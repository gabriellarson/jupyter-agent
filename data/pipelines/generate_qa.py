from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.data import Document
from typing import Iterable
from datatrove.pipeline.base import PipelineStep
import argparse

GENERATE_QA_PROMPT = """# Instructions
You are a data scientist agent.
Your goal is to write down up to 5 QAs about the notebook.
The goal is to write up to 5 most important **complex, multi-step questions** that can be answered by the dataset in the provided notebook. In your answer, do not mention the cell id or reference specific cells. Only talk generally about the notebook and its findings. Only add a question if it is directly answerable by the code execution and the answer is explicitly shown in the notebook outputs, otherwise, do not add it.

## Question Requirements
Questions MUST be:
- **Single, direct questions** that have a specific, measurable answer
- **Complex enough** to require multiple steps of data processing and analysis
- **Answerable with a concrete value, number, or specific finding**
- **Focused exclusively on data analysis and dataset insights** - NOT about machine learning methods, training processes, model parameters, or algorithmic techniques

Each question should require multi-step analysis but result in a **single, direct answer**.

## Answer Requirements
Answers MUST be:
- **Concise and specific** - a direct result that could be numerical, categorical, boolean, or a brief factual statement
- **Not explanatory paragraphs** - just the direct result without elaboration
- **Directly answerable** from the notebook's computational results
- **Explicitly present in the notebook outputs** - do not hallucinate or infer values that are not clearly displayed in the execution results

## Question Complexity Requirements
Questions MUST be **non-trivial** and require **multiple turns of code execution** to answer completely. Each question should involve:

- **Multi-step data analysis**: Questions requiring data preprocessing, transformation, statistical analysis, and interpretation across multiple cells
- **Complex aggregations**: Questions involving grouping, filtering, joining, and advanced statistical computations on the dataset
- **Comparative analysis**: Questions requiring comparisons across multiple dimensions, time periods, or subgroups within the data
- **Statistical inference on data**: Questions involving correlation analysis, hypothesis testing, or statistical relationships within the dataset
- **Derived insights from data**: Questions that cannot be answered by simple data retrieval but require computational analysis and synthesis of dataset information

## Forbidden Question Types
Do NOT include questions about:
- **Machine learning methods**: Training techniques, algorithms, model architectures, hyperparameters
- **Model performance metrics**: Accuracy, loss functions, validation scores, number of parameters
- **Technical implementation**: Code optimization, library usage, computational methods
- **Simple data retrieval**: Single-line operations like "How many rows are in the dataset?"
- **Basic descriptive statistics without context**: "What is the mean of column X?"
- **Questions where answers are not explicitly shown in notebook outputs**

## Required Question Types - Data Analysis Focus
Focus on **Complex Data Analysis Queries** such as:
- Multi-variable trend analysis and patterns within the dataset
- Segmentation analysis involving multiple criteria and data-driven insights
- Performance benchmarking using data metrics and comparative analysis
- Data-driven insights requiring feature analysis and statistical relationships
- Correlation and statistical relationships between variables in the dataset
- Time-series patterns and temporal analysis of data
- Multi-dimensional data optimization and ranking problems
- Data classification or clustering insights with statistical validation

## Quality Control and Fallback
If the notebook does not contain sufficient complexity to generate 5 non-trivial questions requiring multiple code execution steps, OR if the notebook focuses primarily on machine learning training/methodology rather than data analysis, respond with:

**"No QA can be generated for this document"**

Only proceed if the notebook demonstrates:
- Advanced data manipulation and analysis across multiple cells
- Statistical analysis of dataset requiring multiple computational steps
- Complex data visualizations with derived metrics from the dataset
- Multi-stage data processing pipelines focused on data insights
- Clear, explicit outputs that show the results of data analysis

## Critical Requirement: Answer Verification
**MANDATORY**: Every answer must be directly visible and explicitly stated in the notebook's output cells. Do not infer, calculate, or hallucinate any values. If a specific numerical result, category, or finding is not clearly displayed in the execution outputs, do not create a question for it.

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
The answer should be in the following format:
```
<question_1>
# question 1 goes here
</question_1>

<answer_1>
# answer 1 goes here
</answer_1>

<question_2>
# question 2 goes here
</question_2>

<answer_2>
# answer 2 goes here
</answer_2>
```

Here are examples of complex, multi-step data analysis questions with direct answers:
```
<question_1>
What is the highest profit margin percentage achieved by any product category in the dataset?
</question_1>

<answer_1>
72.3%
</answer_1>

<question_2>
Which sales representative achieved the highest quarterly revenue based on the sales data?
</question_2>

<answer_2>
Sarah Johnson
</answer_2>

<question_3>
Is the correlation between marketing spend and sales revenue in the dataset statistically significant?
</question_3>

<answer_3>
Yes
</answer_3>
```

**Why these are good QA examples:** These questions require complex multi-step data analysis (data joining, calculations, statistical testing on the dataset) but result in direct, specific answers that would be explicitly shown in notebook outputs - numerical, categorical, and boolean formats."""


class GenerateQA(PipelineStep):
    name = "❓ Generate QA Pairs"

    def __init__(self, router_ip: str, router_port: int):
        super().__init__()
        self.router_ip = router_ip
        self.router_port = router_port
        self.generate_qa_prompt = GENERATE_QA_PROMPT

    def parse_qa_pairs(self, parsed_content):
        """Parse the qa pairs from the parsed content with question and associated answer"""
        qa_pairs = {}

        for key, value in parsed_content.items():
            if key.startswith("question_"):
                question_num = key.split("_")[1]
                answer_key = f"answer_{question_num}"

                if answer_key in parsed_content:
                    answer_text = parsed_content[answer_key]
                    qa_pairs[question_num] = {"question": value, "answer": answer_text}

        return qa_pairs

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        """Process a single document - this runs in parallel"""
        from .utils.qwen3_formatter_utils import (
            parse_think_and_answer,
            parse_tagged_content,
        )

        from openai import OpenAI

        client = OpenAI(
            api_key="token-abc123",
            base_url=f"http://{self.router_ip}:{self.router_port}/v1",
        )

        for doc in data:
            with self.track_time():
                try:
                    notebook_markdown_content = doc.text["markdown"]  # type: ignore

                    # Generate QAs
                    completion = client.chat.completions.create(
                        model="Qwen/Qwen3-32B",
                        messages=[
                            {"role": "system", "content": self.generate_qa_prompt},
                            {"role": "user", "content": notebook_markdown_content},
                        ],
                        max_completion_tokens=5000,
                    )

                    response_content = completion.choices[0].message.content
                    assert response_content is not None, "response_content is None"
                    think, answer = parse_think_and_answer(response_content)
                    parsed_content = parse_tagged_content(answer)
                    generated_qa_pairs = self.parse_qa_pairs(parsed_content)

                    # Update document metadata
                    doc.metadata["qa_pairs"] = generated_qa_pairs

                except Exception as e:
                    print(f"Error generating QA pairs for document {doc.id}: {e}")
                    return None

            yield doc


class NotebookToMultipleDocuments(PipelineStep):
    """First stage: Split one notebook document into multiple documents (one per Q&A pair)"""

    name = "🔄 NotebookToMultipleDocuments"

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        from datatrove.data import Document
        import json

        for doc in data:
            with self.track_time():
                # Extract metadata and Q&A pairs
                files_used = doc.metadata["files_used"]
                packages_used = doc.metadata["packages_used"]
                qa_pairs = doc.metadata["qa_pairs"]

                # Generate one document per question-answer pair
                for question_num, qa_pair in qa_pairs.items():
                    new_metadata = {
                        **doc.metadata,
                        "files_used": files_used,
                        "packages_used": packages_used,
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "original_document_id": doc.id,
                    }

                    del new_metadata["qa_pairs"]

                    new_doc = Document(
                        text=json.dumps(
                            {
                                "original_notebook": doc.text["notebook"],  # type: ignore
                                "original_markdown": doc.text["markdown"],  # type: ignore
                            }
                        ),  # Clean notebook content
                        id=f"{doc.id}_qa_{question_num}",
                        metadata=new_metadata,
                    )

                    yield new_doc


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

    data_folder = "/fsx/data-agents/data/edu-scoring"
    output_folder = "/fsx/data-agents/data/generate-qa"

    logs_file = "logs/generate-qa"

    pipeline = [
        JsonlReader(data_folder=data_folder, limit=LIMIT),
        # stage 1
        GenerateQA(router_ip=router_ip, router_port=int(router_port)),
        NotebookToMultipleDocuments(),
        JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
    ]

    executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="20:00:00",
        partition="hopper-prod",
        logging_dir=logs_file,
        cpus_per_task=4,
        mem_per_cpu_gb=16,
        qos="high",
        skip_completed=False,
        job_name="qa-generation",
    )

    executor.run()


if __name__ == "__main__":
    main()
