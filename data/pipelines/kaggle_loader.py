from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from typing import Iterable
from datatrove.data import Document


class KaggleLoader(PipelineStep):
    name = "🏁 KaggleLoader"

    def __init__(self, base_path: str, limit: int = -1):
        super().__init__()
        self.base_path = base_path
        self.limit = limit

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        import zipfile
        from datatrove.data import Document

        file_exclusion_list = [
            "0132/087/132087938.ipynb"  # file that always caused a crash
        ]

        # there are 20 files with 10 subfolders each except the last one only has one --> 191 folders we distribute across ranks
        filepath = f"{self.base_path}/metakaggle_code_with_outputs_{str(rank // 10).zfill(3)}.zip"

        documents_processed = 0  # Counter for processed documents

        with zipfile.ZipFile(filepath, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                # Check if we've reached the limit (if limit is set)
                if self.limit != -1 and documents_processed >= self.limit:
                    print(
                        f"Reached limit of {self.limit} documents. Stopping processing."
                    )
                    break

                # check if it is a file and it matches the pattern of current rank (skip broken files)
                if not file_info.is_dir() and file_info.filename.split("/")[0] == str(
                    rank
                ).zfill(4):
                    if file_info.filename in file_exclusion_list:
                        print(f"Skipped: {file_info.filename}")
                        continue
                    print(file_info.filename)

                    # try to read file and add it as raw text file
                    with zip_ref.open(file_info.filename) as file:
                        try:
                            content = file.read().decode("utf-8")
                        except Exception as e:
                            print(f"Error decoding: {file_info.filename}: {e}")
                            continue

                    document = Document(text=content, id=file_info.filename)
                    documents_processed += 1  # Increment counter
                    yield document


def main():
    N_TASKS = 191
    N_WORKERS = 191
    LIMIT = -1
    BASE_PATH = "/fsx/jupyter-agent/data/meta-kaggle-dataset"

    logs_file = "logs/kaggle-loader"
    output_folder = "/fsx/jupyter-agent/data/meta-kaggle-dataset-datatrove-format"

    executor = SlurmPipelineExecutor(
        pipeline=[
            KaggleLoader(base_path=BASE_PATH, limit=LIMIT),
            JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=logs_file,
        cpus_per_task=4,
        mem_per_cpu_gb=4,
        qos="high",
        skip_completed=False,
        job_name="kaggle-loader",
    )

    executor.run()


if __name__ == "__main__":
    main()
