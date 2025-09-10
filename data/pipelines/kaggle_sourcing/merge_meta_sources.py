import kagglehub
import pandas as pd
import os
from pandarallel import pandarallel

path = kagglehub.dataset_download("kaggle/meta-kaggle")
print("Path to dataset files:", path)

kernel_dataset_version_sources = pd.read_csv(
    os.path.join(path, "KernelVersionDatasetSources.csv"),
    usecols=["Id", "KernelVersionId", "SourceDatasetVersionId"],
)
# rename Id to KernelId for clarity
kernel_dataset_version_sources.rename(columns={"Id": "KernelId"}, inplace=True)

dataset_versions = pd.read_csv(
    os.path.join(path, "DatasetVersions.csv"),
    usecols=["DatasourceVersionId", "DatasetId", "Slug"],
)
datasets = pd.read_csv(
    os.path.join(path, "Datasets.csv"),
    usecols=["Id", "OwnerUserId", "OwnerOrganizationId"],
)
users = pd.read_csv(os.path.join(path, "Users.csv"), usecols=["Id", "UserName"])
organizations = pd.read_csv(
    os.path.join(path, "Organizations.csv"), usecols=["Id", "Slug"]
)


print("Kernel Dataset Version Sources shape:", kernel_dataset_version_sources.shape)
print("Dataset Versions shape:", dataset_versions.shape)
print("Datasets shape:", datasets.shape)
print("Users shape:", users.shape)
print("Organizations shape:", organizations.shape)

merged1 = kernel_dataset_version_sources.merge(
    dataset_versions,
    left_on="SourceDatasetVersionId",
    right_on="DatasourceVersionId",
    how="left",
)

print(
    "Merged Kernel Dataset Version Sources with Dataset Versions shape:", merged1.shape
)

merged2 = merged1.merge(datasets, left_on="DatasetId", right_on="Id", how="left")

print("Merged with Datasets shape:", merged2.shape)

merged2["FullDatasetName"] = merged2["Slug"]


def get_author_name(row):
    if not pd.isna(row["OwnerUserId"]) and row["OwnerUserId"] != "":
        user = users.loc[users["Id"] == row["OwnerUserId"], "UserName"]
        return user.values[0] if not user.empty else None
    elif not pd.isna(row["OwnerOrganizationId"]) and row["OwnerOrganizationId"] != "":
        org = organizations.loc[
            organizations["Id"] == row["OwnerOrganizationId"], "Slug"
        ]
        return org.values[0] if not org.empty else None
    else:
        return None


pandarallel.initialize(nb_workers=32, progress_bar=True)

merged2["AuthorName"] = merged2.parallel_apply(get_author_name, axis=1)


def make_full_path(row):
    if row["AuthorName"] is not None and row["FullDatasetName"] is not None:
        return f"{row['AuthorName']}/{row['FullDatasetName']}"
    else:
        return ""


merged2["FullDatasetPath"] = merged2.parallel_apply(make_full_path, axis=1)

final_df = merged2[
    ["KernelId", "KernelVersionId", "FullDatasetPath", "FullDatasetName", "AuthorName"]
]

print("Final DataFrame shape:", final_df.shape)

final_df[:100].to_csv("/fsx/jupyter-agent/data/kaggle-sourcing/final_merged_demo.csv", index=False)

final_df.to_csv("/fsx/jupyter-agent/data/kaggle-sourcing/merged_metadata.csv", index=False)
