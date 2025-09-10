from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datasets import load_dataset
from typing import Optional


class Dedup(BaseFilter):
    name = "🕵 Dedup"

    def __init__(self, exclusion_writer: Optional[DiskWriter] = None):
        """
        filter files based on a filelist

        Args:
            filelist: path to filelist
            exclusion_writer:
        """
        super().__init__(exclusion_writer)  # type: ignore

        ds = load_dataset("bigcode/starcoder2data-extras", "kaggle", split="train")
        self.filelist = list(ds["file_id"])

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        return doc.id.split("/")[-1].split(".")[0] in self.filelist


def main():
    N_TASKS = 128
    N_WORKERS = 128
    logs_file = "logs/kaggle-filter"
    data_folder = "/fsx/jupyter-agent/data/meta-kaggle-dataset-datatrove-format"
    output_folder = "/fsx/jupyter-agent/data/filtered"

    executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(data_folder=data_folder),
            # The dataset has already been cleaned and deduplicated as part of the BigCode project
            # Stage 1: Filter out files that are not in the BigCode filelist (i.e that are duplicates)
            Dedup(),
            JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=logs_file,
        cpus_per_task=4,
        mem_per_cpu_gb=16,
        qos="high",
        skip_completed=False,
        job_name="dedup",
    )

    executor.run()


if __name__ == "__main__":
    main()
