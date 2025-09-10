import argparse
from typing import Iterable, Optional
from .add_markdown_to_notebook.add_markdown_to_notebook import AddMarkdownToNotebook

from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.base import PipelineStep

from datatrove.pipeline.writers.disk_base import DiskWriter

TEMPLATE = """
Below is an extract from a Jupyter notebook. Evaluate whether it has a high analysis value and could help a data scientist. 

# Notebook format
The notebooks are formatted with the following tokens:

<|start_markdown_cell|>
(Here comes markdown content)
</|end_markdown_cell|>

<|start_python_cell|>
(Here comes python code)
</|end_python_cell|>

<|start_output_cell|>
(Here comes code output of the previous python cell (truncated to first 25 lines if longer)
If output is longer than 25 lines, you'll see "[Output is truncated as it is more than 25 lines]")
</|end_output_cell|>

# Scoring system
Below is an extract from a Jupyter notebook. Evaluate its quality for data analysis and explanotary value using a 5-point scoring system.

SCORING CRITERIA (additive, stop if criterion fails):
1. **Valid Python Code** (1 point): Contains syntactically correct Python code that could execute without errors.
2. **Data Loading** (1 point): Successfully loads and displays a dataset (CSV, JSON, API, etc.) with visible output confirming the load.
3. **Data Analysis** (1 point): Performs meaningful analysis beyond basic loading - statistics, transformations, filtering, or visualizations with appropriate outputs.
4. **Educational Narrative** (1 point): Majority of code cells are accompanied by markdown explanations that interpret results, explain methodology, or provide insights (not just code comments).
5. **Exceptional Quality** (1 point): Demonstrates advanced analysis with:
   - Multiple interconnected analytical steps
   - Well-designed, annotated visualizations
   - Clear problem-solving progression
   - Actionable insights or conclusions

Only exceptional notebooks meeting all 5 criteria above get the full 5 points. They should be above the level of student exercies or introductory material on sample datasets.

EXCLUSIONS (automatic 0 points):
- Non-English text
- Non-Python code
- No data analysis component
- Only text cells OR only code cells
- Broken/incomplete code blocks

Provide:
1. Brief justification (max 80 words)
2. Educational score: <number>

Here is the extract:
{}
"""


class EduScorer(PipelineStep):
    name = "👨‍🎓 EduScorer"

    def __init__(self, router_ip: str, router_port: str):
        super().__init__()
        self.router_ip = router_ip
        self.router_port = router_port
        self.template = TEMPLATE

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        import re
        from openai import OpenAI

        client = OpenAI(
            api_key="token-abc123",
            base_url=f"http://{self.router_ip}:{self.router_port}/v1",
        )

        for doc in data:
            with self.track_time():
                try:
                    prompt = self.template.format(doc.text["markdown"])  # type: ignore

                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="Qwen/Qwen3-32B",
                        max_completion_tokens=1000,
                    )

                    response = chat_completion.choices[0].message.content

                    assert response is not None, "Response is None"

                    match = re.search(
                        r"^\D*(\d)", response.split("Educational score:")[-1].strip()
                    )
                    score = match.group(1) if match else "no match"

                    # Assign label based on score
                    label = (
                        int(score) if score in ["0", "1", "2", "3", "4", "5"] else -1
                    )

                    doc.metadata["edu_score"] = label
                    doc.metadata["edu_error"] = ""

                except Exception as e:
                    doc.metadata["edu_score"] = -1
                    doc.metadata["edu_error"] = str(e)

            yield doc


class EduFilter(BaseFilter):
    name = "🕵 EduFilter"

    def __init__(
        self,
        threshold: int = 4,
        exclusion_writer: Optional[DiskWriter] = None,
        batch_size: int = 1,
    ):
        """
        filter files based on edu_score

        Args:
            threshold: threshold for edu_score
        """
        super().__init__(exclusion_writer=exclusion_writer, batch_size=batch_size)  # type: ignore
        self.threshold = threshold

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter, bool: True if the document should be kept
        """

        return doc.metadata["edu_score"] >= self.threshold


def main():
    N_TASKS = 128
    N_WORKERS = 128
    LIMIT = -1
    logs_file = "logs/edu-scoring"
    data_folder = "/fsx/jupyter-agent/data/filtered"
    output_folder = "/fsx/jupyter-agent/data/edu-scoring"

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

    executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(data_folder=data_folder, limit=LIMIT),
            # Stage 1: add markdown to notebook (for LLM to process)
            AddMarkdownToNotebook(),
            # Stage 2: score each notebook
            EduScorer(router_ip=router_ip, router_port=router_port),
            # Stage 3: filter out notebooks with low edu_score
            EduFilter(threshold=4),
            JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
        ],
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="4-00:00:00",
        partition="hopper-cpu",
        logging_dir=logs_file,
        cpus_per_task=4,
        mem_per_cpu_gb=16,
        qos="normal",
        skip_completed=False,
        job_name="edu-scoring",
    )

    executor.run()


if __name__ == "__main__":
    main()
