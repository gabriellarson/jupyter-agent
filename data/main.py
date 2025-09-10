from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor

from pipelines.add_markdown_to_notebook.add_markdown_to_notebook import (
    AddMarkdownToNotebook,
)
from pipelines.edu_scoring import EduScorer
from pipelines.extract_packages_and_files import (
    ExtractPackagesAndFiles,
)
from pipelines.generate_qa import GenerateQA
from pipelines.dedup import Dedup
from pipelines.kaggle_loader import KaggleLoader
from pipelines.kaggle_data_mapping import (
    KaggleDatasetMapper,
    DataSourceMapper,
    DatasetMapFilter,
    KaggleDatasetDownloader,
)
from pipelines.generate_traces import GenerateTraces
from pipelines.convert_to_chatml_format import DataAgentTracesNotebookToChatML
import argparse


def main():
    # To run the pipeline you need a OpenAPI compatible endpoint
    parser = argparse.ArgumentParser(description="Run the data processing pipeline")

    parser.add_argument(
        "-r",
        "--router",
        type=str,
        required=True,
        help="Router address in format IP or IP:PORT (default port 39876 if not specified)",
    )
    args = parser.parse_args()

    if ":" in args.router:
        router_ip, router_port = args.router.split(":")
    else:
        router_ip = args.router
        router_port = "39876"

    assert router_ip is not None, "router_ip is required"
    assert router_port is not None, "router_port is required"

    N_TASKS = 1
    N_WORKERS = 1
    LIMIT = -1

    BASE_PATH = "/fsx/jupyter-agent/data/meta-kaggle-dataset"

    output_folder = "/fsx/jupyter-agent/data/traces-final"

    logs_file = "logs/jupyter-agent-data-pipeline"

    executor = SlurmPipelineExecutor(
        pipeline=[
            # Stage 0: load the kaggle dataset
            KaggleLoader(base_path=BASE_PATH, limit=LIMIT),
            # # Stage 1: Dedup the dataset (using our work in BigCode where we dedup kaggle meta dataset)
            Dedup(),
            # Todo to remove, but allow for easier debugging because stage 1 dedup clean all the docs -> goes from 2tb to 200GB so lot of removing
            # JsonlReader(data_folder="s3://data-agents/kaggle-outputs/filtered-with-output", limit=LIMIT),
            # Stage 2: Edu scoring
            AddMarkdownToNotebook(),
            EduScorer(router_ip=router_ip, router_port=router_port),
            # EduFilter(threshold=4),
            # Stage 3: Extract packages and files used (remove notebooks that don't use any files as we are interested in notebook that do data analysis)
            ExtractPackagesAndFiles(router_ip=router_ip, router_port=router_port),
            # RemoveNoFilesUsed(),
            # Stage 4: Generate QA pairs
            GenerateQA(router_ip=router_ip, router_port=int(router_port)),
            # Stage 5: Get original Kaggle datasets and map them to notebooks that were used to generate the QA pairs
            KaggleDatasetMapper(),
            DataSourceMapper(),
            DatasetMapFilter(),
            KaggleDatasetDownloader(),
            # Stage 6: Generate synthetic code and traces with code execution for the QA pairs generated in stage 4
            GenerateTraces(router_ip=router_ip, router_port=router_port),
            DataAgentTracesNotebookToChatML(),
            # Write down the dataset
            JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=logs_file,
        cpus_per_task=1,
        mem_per_cpu_gb=4,
        qos="high",
        skip_completed=False,
        job_name="jupyter-agent-data-pipeline",
    )

    executor.run()


if __name__ == "__main__":
    main()
