from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document
import os
import pandas as pd
from typing import Iterable


from datetime import datetime
from zoneinfo import ZoneInfo

os.environ["KAGGLEHUB_CACHE"] = "/fsx/jupyter-agent/kaggle-datasets"
os.environ["KAGGLE_USERNAME"] = "<optional>"
os.environ["KAGGLE_KEY"] = "<optional>"


class KaggleDatasetMapper(PipelineStep):
    name = "📖 KaggleDatasetMapper"

    def __init__(self):
        super().__init__()
        self.mapping_df = pd.read_csv("/fsx/jupyter-agent/data/kaggle-sourcing/merged_metadata.csv")

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        import pandas as pd
        keywords = [
            "model",
            "vit-",
            "resnet",
            "alexnet",
            "vgg",
            "torch",
            "keras",
            "-tf",
            "tensorflow",
            "gemma",
            "-llama",
            "mistral",
            "bert",
            "llm-",
            "-llm",
            "clip",
            "laion",
            "2b-",
            "7b-",
            "8b-",
            "3b-",
            "pnw-mountains",
        ]

        kernelid_to_dataset = dict(
            zip(
                self.mapping_df["KernelVersionId"].astype(str),
                self.mapping_df["FullDatasetPath"],
            )
        )
        for doc in data:
            with self.track_time():
                doc_id = doc.id.split("/")[-1].split(".ipynb")[0]
                doc.metadata["kaggle_dataset_name"] = None
                dataset_name = kernelid_to_dataset.get(doc_id, None)
                if pd.isna(dataset_name):
                    dataset_name = None
                elif not any(sw in dataset_name for sw in keywords):
                    doc.metadata["kaggle_dataset_name"] = dataset_name

            yield doc


class DataSourceMapper(PipelineStep):
    name = "📖 DataSourceMapper"

    def __init__(self):
        super().__init__()

    def find_url_dataset(self, doc: Document):
        import re
        import json

        notebook_dict = json.loads(doc.text)

        download_keywords = [
            "wget",
            "curl",
            "requests.get",
            "urllib.request",
            "download",
            "pd.read_csv",
            "pd.read_json",
            "read_csv",
            "read_json",
        ]
        for cell in notebook_dict.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                # Only consider code that contains download keywords
                if any(kw in source for kw in download_keywords):
                    found_urls = re.findall(r"https?://\S+", source)
                    if found_urls:
                        return True
        return False

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        for doc in data:
            with self.track_time():
                if doc.metadata.get("kaggle_dataset_name") is not None:
                    doc.metadata["data_source_type"] = "kaggle"
                elif self.find_url_dataset(doc):
                    doc.metadata["data_source_type"] = "url"
                elif doc.metadata.get("files_used") is not None:
                    doc.metadata["data_source_type"] = "local_file"
                else:
                    doc.metadata["data_source_type"] = None

            yield doc


class DatasetMapFilter(BaseFilter):
    name = "📖 DatasetMapFilter"

    def __init__(self, threshold: int = 4):
        """
        filter notebooks in python language

        Args:
            filelist: path to filelist
            exclusion_writer:
        """
        super().__init__()
        self.threshold = threshold

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter, bool: True if the document should be kept
        """

        if doc.metadata.get("data_source_type") is not None:
            if (
                doc.metadata.get("data_source_type") == "kaggle"
                and len(doc.metadata["files_used"]) > 0
            ):  # use this line for QA dataset
                # if doc.metadata.get('data_source_type')  == 'kaggle': # use this line for files without metadata
                return True
            elif doc.metadata.get("data_source_type") == "url":
                return True

        return False


class KaggleDatasetDownloader(PipelineStep):
    name = "📖 KaggleDatasetDownloader"

    def __init__(self):
        super().__init__()

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        import kagglehub
        import time
        import os

        keywords = [
            "model",
            "vit-",
            "resnet",
            "alexnet",
            "vgg",
            "torch",
            "keras",
            "-tf",
            "tensorflow",
            "gemma",
            "-llama",
            "mistral",
            "bert",
            "llm-",
            "-llm",
        ]

        for doc in data:
            dataset_name = doc.metadata.get("kaggle_dataset_name", None)
            # Skip download if any stopword is in dataset_name
            if dataset_name is not None and not any(
                sw in dataset_name for sw in keywords
            ):
                dataset_base = (
                    f"/fsx/jupyter-agent/kaggle-datasets/datasets/{dataset_name}"
                )
                versions_dir = os.path.join(dataset_base, "versions")
                version_path = None
                if os.path.exists(versions_dir):
                    # Find the highest-numbered folder inside versions
                    version_folders = [
                        d
                        for d in os.listdir(versions_dir)
                        if d.isdigit() and os.path.isdir(os.path.join(versions_dir, d))
                    ]
                    if version_folders:
                        highest_version = max(version_folders, key=lambda x: int(x))
                        version_path = os.path.join(versions_dir, highest_version)
                if version_path and os.path.exists(version_path):
                    doc.metadata["kaggle_dataset_path"] = version_path
                    print(
                        f"Dataset {doc.metadata['kaggle_dataset_path']} already exists, skipping download."
                    )
                elif os.path.exists(dataset_base):
                    doc.metadata["kaggle_dataset_path"] = dataset_base
                    print(
                        f"Dataset {doc.metadata['kaggle_dataset_path']} already exists, skipping download."
                    )
                else:
                    n_tries = 0
                    max_tries = 3
                    path = None
                    while n_tries < max_tries and path is None:
                        n_tries += 1
                        try:
                            path = kagglehub.dataset_download(dataset_name)
                            print(path)
                            doc.metadata["kaggle_dataset_path"] = path
                        except Exception as e:
                            print(f"Error downloading {dataset_name}: {e}")
                            time.sleep(60)
                            doc.metadata["kaggle_dataset_path"] = None
            else:
                doc.metadata["kaggle_dataset_path"] = None

            yield doc


def main():
    N_TASKS = 4
    N_WORKERS = 64
    LIMIT = -1

    # Get current time in Amsterdam timezone
    amsterdam_time = datetime.now(ZoneInfo("Europe/Amsterdam"))
    timestamp = amsterdam_time.strftime("%d-%m-%Y_%H:%M:%S")

    logs_file = f"./logs/dataset-downloading/{timestamp}"
    data_folder = "/fsx/jupyter-agent/data/generate-qa"
    output_folder = "/fsx/jupyter-agent/data/kaggle-mapped-test"

    logs_file = "logs/kaggle-data-sourcing"

    executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(data_folder=data_folder, limit=LIMIT),
            KaggleDatasetMapper(),
            DataSourceMapper(),
            DatasetMapFilter(),
            KaggleDatasetDownloader(),
            JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-prod",
        logging_dir=logs_file,
        cpus_per_task=4,
        mem_per_cpu_gb=16,
        qos="high",
        skip_completed=False,
    )

    executor.run()


if __name__ == "__main__":
    main()
