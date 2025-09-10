from typing import Iterable
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.readers import JsonlReader
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.base import PipelineStep
import argparse

FILES_USED_PROMPT = """# Instructions
You are a helpful coding agent.
Your goal is to write down a list of files that are used in the notebook.

# Notebook format
The format of the notebook is the following:
```
<markdown_cell>
<id:cell_0>
# markdown code goes here
</markdown_cell>

<python_cell>
<id:cell_1>
# python code goes here
</python_cell>

<output_cell>
<id:cell_1>
# output of the python code goes here
</output_cell>
```

For example:
```
<markdown_cell>
<id:cell_0>
# This is a hello world notebook
</markdown_cell>

<python_cell>
<id:cell_1>
def hello_world():
    print("Hello, World!")
</python_cell>

<output_cell>
<id:cell_1>
Hello, World!
</output_cell>
```

# Output format
The answer should be in the following format:
```
<files_used>
# files used goes here
</files_used>
```

for example:
```
<files_used>
- ../data/train.csv
- test.csv
- sample_submission.csv
</files_used>
```"""

PACKAGES_USED_PROMPT = """# Instructions
You are a helpful coding agent.
Your goal is to write down a list of packages that are used in the notebook.

# Notebook format
The format of the notebook is the following:
```
<markdown_cell>
<id:cell_1>
# markdown code goes here
</markdown_cell>

<python_cell>
<id:cell_2>
# python code goes here
</python_cell>

<output_cell>
<id:cell_2>
# output of the python code goes here
</output_cell>
```

For example:
```
<markdown_cell>
<id:cell_1>
# This is a hello world notebook
</markdown_cell>

<python_cell>
<id:cell_2>
def hello_world():
    print("Hello, World!")
</python_cell>

<output_cell>
<id:cell_2>
Hello, World!
</output_cell>
```

# Output format
Your answer should be in the following format:
```
<packages_used>
# packages used goes here
</packages_used>
```

for example:
```
<packages_used>
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- xgboost
- lightgbm
</packages_used>
```"""


class ExtractPackagesAndFiles(PipelineStep):
    name = "📦 Extract Packages and Files Used"

    def __init__(self, router_ip: str, router_port: str):
        super().__init__()
        self.router_ip = router_ip
        self.router_port = router_port
        self.files_used_prompt = FILES_USED_PROMPT
        self.packages_used_prompt = PACKAGES_USED_PROMPT

    def parse_list_content(self, content_str):
        """Parse list content from string format to actual list"""
        import re

        # Split by newlines and clean up
        lines = content_str.strip().split("\n")
        parsed_list = []

        for line in lines:
            line = line.strip()
            if line:
                # Remove leading dashes, bullets, or numbers
                cleaned_line = re.sub(r"^[-*•]\s*", "", line)
                cleaned_line = re.sub(r"^\d+\.\s*", "", cleaned_line)
                if cleaned_line:
                    parsed_list.append(cleaned_line)

        return parsed_list

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        """Process a single document - this runs in parallel"""

        from openai import OpenAI

        client = OpenAI(
            api_key="token-abc123",
            base_url=f"http://{self.router_ip}:{self.router_port}/v1",
        )

        from .utils.qwen3_formatter_utils import (
            parse_think_and_answer,
            parse_tagged_content,
        )

        for doc in data:
            with self.track_time():
                try:
                    notebook_markdown_content = doc.text["markdown"]  # type: ignore

                    # Generate files used
                    completion = client.chat.completions.create(
                        model="Qwen/Qwen3-32B",
                        messages=[
                            {"role": "system", "content": self.files_used_prompt},
                            {"role": "user", "content": notebook_markdown_content},
                        ],
                        max_completion_tokens=1000,
                    )

                    response_content = completion.choices[0].message.content
                    assert response_content is not None, "response_content is None"
                    think, answer = parse_think_and_answer(response_content)
                    parsed_content = parse_tagged_content(answer)
                    generated_files_used = self.parse_list_content(
                        parsed_content["files_used"]
                    )

                    # Generate packages used
                    completion = client.chat.completions.create(
                        model="Qwen/Qwen3-32B",
                        messages=[
                            {"role": "system", "content": self.packages_used_prompt},
                            {"role": "user", "content": notebook_markdown_content},
                        ],
                        max_completion_tokens=1000,
                    )

                    response_content = completion.choices[0].message.content
                    assert response_content is not None, "response_content is None"
                    think, answer = parse_think_and_answer(response_content)
                    parsed_content = parse_tagged_content(answer)
                    generated_packages_used = self.parse_list_content(
                        parsed_content["packages_used"]
                    )

                    # Update document metadata
                    doc.metadata["files_used"] = generated_files_used
                    doc.metadata["packages_used"] = generated_packages_used

                except Exception as e:
                    print(
                        f"Error generating files and packages metadata for document {doc.id}: {e}"
                    )
                    return None

            yield doc


class RemoveNoFilesUsed(BaseFilter):
    name = "🚫 RemoveNoFilesUsed"

    def __init__(self, max_turns: int = 10):
        super().__init__()
        self.max_turns = max_turns

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter, bool: True if the document should be kept (no more than max_turns turns)
        """

        files_used = doc.metadata["files_used"]

        if len(files_used) == 0:
            print("Removing document with no files used")
            return False
        return True


def main():
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

    N_TASKS = 128
    N_WORKERS = 128
    LIMIT = -1

    data_folder = "/fsx/jupyter-agent/data/edu-scoring"
    output_folder = "/fsx/jupyter-agent/data/extract-packages-and-files"
    logs_file = "logs/extract-packages-and-files"

    executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(data_folder=data_folder, limit=LIMIT),
            # stage 1: Extract packages and files used
            ExtractPackagesAndFiles(router_ip=router_ip, router_port=router_port),
            # stage 2: Remove notebooks that don't use any files
            RemoveNoFilesUsed(),
            JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-cpu",
        logging_dir=logs_file,
        cpus_per_task=4,
        mem_per_cpu_gb=16,
        qos="normal",
        skip_completed=False,
        job_name="metadata-generation",
    )

    executor.run()


if __name__ == "__main__":
    main()
