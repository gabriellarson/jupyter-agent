from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.data import Document
from pipelines.convert_to_chatml_format import DataAgentTracesNotebookToChatML
from pipelines.utils.qwen3_formatter_utils import load_markdown_file
from typing import Iterable


class GenerateTraces(PipelineStep):
    name = "🧠 GenerateTracesFromQuestion"

    def __init__(self, router_ip: str, router_port: str):
        super().__init__()
        self.router_ip = router_ip
        self.router_port = router_port
        self.max_turns = 25

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        """Process a single document - this runs in parallel"""
        import json
        from pipelines.utils.qwen3_formatter_utils import (
            extract_original_notebook_content,
            tools_thinking,
            tools_non_thinking,
            load_markdown_file,
        )
        from pipelines.utils.jupyter_utils import (
            create_base_notebook,
            assistant_thinking_template,
            assistant_final_answer_template,
            extract_code_from_block,
        )
        from pipelines.utils.sandbox import Sandbox, SandboxE2B
        import e2b_code_interpreter as e2b
        import os
        import random
        import time
        from openai import OpenAI

        client = OpenAI(
            api_key="token-abc123",
            base_url=f"http://{self.router_ip}:{self.router_port}/v1",
        )

        for doc in data:
            try:
                # Extract Q&A and metadata from document
                if doc.metadata["question"] is None:
                    print(
                        f"Skipping document {doc.id} as it has no question in metadata."
                    )
                    return None

                # Extract original notebook content (remove Q&A metadata)
                doc_text_data = doc.text
                # or use this depending on whether you use QA data source or not
                doc_text_data = json.loads(doc.text)
                original_notebook = extract_original_notebook_content(
                    doc_text_data["original_notebook"]
                )

                # build system prompt if qwen coder is used
                # model_gen_name = "Qwen/Qwen3-32B" # uncomment to use thinking model
                model_gen_name = "Qwen3-Coder-480B-A35B-Instruct"

                if "Coder" in model_gen_name:
                    system_prompt_coder = {
                        "role": "system",
                        "content": "Always make a step-by-step plan before running tool calls and start every message with 'Ok here's the plan:'. Always write at least 5 sentences of what you observed and what you want to do next. For every code action write down your throught process and what are you doing. Make sure you provide some message content *not* in the code, but for the markdown block. Always try to call a tool if available and do not get stuck on generating the repetitive code.",
                    }
                    tools = tools_non_thinking
                else:
                    system_prompt_coder = {
                        "role": "system",
                        "content": "Always try to call a tool if available and do not get stuck on generating the repetitive code.",
                    }
                    tools = tools_thinking

                user_prompt_path = "./pipelines/prompts/agent_prompt_llm.md"
                user_prompt_path_e2b = "./pipelines/prompts/agent_prompt_e2b.md"

                def get_valid_filename(file):
                    # Take last part after '/' if present, else whole string, and ensure not empty
                    fname = file.split("/")[-1] if "/" in file else file
                    return fname if fname and len(fname) > 0 else file

                # check if e2b can be used
                if (
                    doc.metadata["data_source_type"] == "kaggle"
                    or doc.metadata["data_source_type"] == "url"
                ):
                    type_executor = "e2b"
                else:
                    type_executor = "llm"

                # check if the data files exist locally
                if (
                    doc.metadata["data_source_type"] == "kaggle"
                    and len(doc.metadata["files_used"]) > 0
                ):
                    for file in doc.metadata["files_used"]:
                        if not os.path.exists(
                            os.path.join(
                                doc.metadata["kaggle_dataset_path"],
                                get_valid_filename(file),
                            )
                        ):
                            # print(f"File {file} | {get_valid_filename(file)} does not exist in {doc.metadata['kaggle_dataset_path']}, using LLM executor.")
                            type_executor = "llm"

                question = doc.metadata["question"]
                files_used_formatted_str_llm = "\n".join(
                    f"- {file}" for file in doc.metadata["files_used"]
                )
                files_used_formatted_str_e2b = "\n".join(
                    f"- {get_valid_filename(file)}"
                    for file in doc.metadata["files_used"]
                )
                preinstalled_e2b_packages = [
                    "aiohttp",
                    "beautifulsoup4",
                    "bokeh",
                    "gensim",
                    "imageio",
                    "joblib",
                    "librosa",
                    "matplotlib",
                    "nltk",
                    "numpy",
                    "opencv-python",
                    "openpyxl",
                    "pandas",
                    "plotly",
                    "pytest",
                    "python-docx",
                    "pytz",
                    "requests",
                    "scikit-image",
                    "scikit-learn",
                    "scipy",
                    "seaborn",
                    "soundfile",
                    "spacy",
                    "textblob",
                    "tornado",
                    "urllib3",
                    "xarray",
                    "xlrd",
                    "sympy",
                ]
                packages_original = doc.metadata.get("packages_used", None)
                packages_used = []
                for package in packages_original:
                    if (
                        "." in package and "mpl" not in package
                    ):  # some packages are in the format 'package.subpackage', we only need package name
                        packages_used.append(package.split(".")[0])
                    elif (
                        "mpl" in package
                    ):  # mpl_tools is bugged in metadata, overwriting to the correct name
                        packages_used.append("mpl-tools")
                    elif package == "os":
                        continue  # skip os as it does not exist in pip
                    else:
                        packages_used.append(package)
                packages_used.append(
                    "tabulate"
                )  # tabulate is often used by LLMs to format output tables and looks nicer
                packages_used.append(
                    "statsmodels"
                )  # statsmodels is often used by LLMs for statistical modeling
                # Remove packages that are already preinstalled in e2b if using e2b executor
                if type_executor == "e2b":
                    packages_used = [
                        pkg
                        for pkg in packages_used
                        if pkg not in preinstalled_e2b_packages
                    ]
                packages_used_bullet_str = (
                    "\n".join(f"- {pkg}" for pkg in packages_used)
                    if packages_used
                    else "No information provided."
                )

                if type_executor == "llm":
                    agent_prompt = load_markdown_file(user_prompt_path).format(
                        files=files_used_formatted_str_llm,
                        question=question,
                        packages=packages_used_bullet_str,
                    )
                else:
                    # For E2B, we use a different prompt that includes uses different filename system to enable local file access
                    agent_prompt = load_markdown_file(user_prompt_path_e2b).format(
                        files=files_used_formatted_str_e2b,
                        question=question,
                        packages=packages_used_bullet_str,
                    )

                messages = [
                    system_prompt_coder,
                    {"role": "user", "content": agent_prompt},
                ]

                notebook_data, _ = create_base_notebook(messages)

                # explore and load data first to give better context to the model for kaggle or url based datasets
                if type_executor == "e2b":
                    # if the data source is kaggle or url, we need to load the data first
                    # and then generate the code to explore the data
                    files_used_formatted_str = "\n".join(
                        f"- {get_valid_filename(file)}"
                        for file in doc.metadata["files_used"]
                    )  # DO NOT include full parsed data file path of the model as we do not have it
                    data_eda_prompt = load_markdown_file(
                        "./pipelines/prompts/data_exploration_prompt.md"
                    ).format(
                        files=files_used_formatted_str,
                        original_notebook=original_notebook,
                        question=question,
                        packages=packages_used_bullet_str,
                    )

                    messages_data_eda = [
                        system_prompt_coder,
                        {"role": "user", "content": data_eda_prompt},
                    ]

                    eda_code = None
                    while eda_code is None:
                        try:
                            response = client.chat.completions.create(
                                messages=messages_data_eda,
                                model=model_gen_name,
                                tools=tools,
                            )
                            full_response = response.choices[0].message.content
                            if response.choices[0].message.tool_calls:
                                tool_call = (
                                    response.choices[0].message.tool_calls[0] or []
                                )
                                tool_args = json.loads(tool_call.function.arguments)
                            if "Coder" in model_gen_name:
                                if tool_call and "jupyter" in tool_call.function.name:
                                    if tool_args["comment"]:
                                        full_response = (
                                            "<think>"
                                            + tool_args["comment"]
                                            + "</think>"
                                        )
                                elif (
                                    response.choices[0].message.content is not None
                                    and len(response.choices[0].message.content) > 0
                                ):
                                    full_response = (
                                        "<think>"
                                        + response.choices[0].message.content
                                        + "</think>"
                                    )
                                else:
                                    full_response = (
                                        "<think>Executing next step...</think>"
                                    )
                            else:
                                full_response = response.choices[0].message.content

                            tool_args = json.loads(tool_call.function.arguments)
                            eda_code = tool_args["code"]

                        except Exception as e:
                            print(f"Error: {e} for messages: {messages_data_eda}")
                            continue

                    sandbox_init = None
                    while sandbox_init is None:
                        try:
                            # Initialize E2B sandbox
                            time.sleep(
                                random.randrange(1, 5)
                            )  # random delay to avoid E2B rate limit
                            sandbox_init = e2b.Sandbox(timeout=720)
                        except Exception as e:
                            print(f"Error initializing E2B sandbox: {e}")
                            time.sleep(10)

                    if doc.metadata["kaggle_dataset_path"] is not None:
                        file_error = False
                        for file_name in doc.metadata["files_used"]:
                            if os.path.exists(
                                os.path.join(
                                    doc.metadata["kaggle_dataset_path"],
                                    get_valid_filename(file_name),
                                )
                            ):
                                with open(
                                    f"{doc.metadata['kaggle_dataset_path']}/{get_valid_filename(file_name)}",
                                    "rb",
                                ) as file:
                                    try:
                                        sandbox_init.files.write(
                                            f"/home/user/input/{get_valid_filename(file_name)}",
                                            file,
                                        )
                                    except Exception as e:
                                        print(
                                            f"Error writing file {file_name} to sandbox: {e}"
                                        )
                                        # Switch to LLM executor and skip e2b loop; sometimes writing big files causes a WriteError
                                        type_executor = "llm"
                                        agent_prompt = load_markdown_file(
                                            user_prompt_path
                                        ).format(
                                            files=files_used_formatted_str_llm,
                                            question=question,
                                            packages=packages_used_bullet_str,
                                        )
                                        file_error = True
                                        break
                            if file_error:
                                # Skip the rest of the e2b loop if we cannot upload file and proceed with LLM executor
                                pass

                    sandbox_eda = SandboxE2B(original_notebook, sandbox_init, doc)

                    if type_executor == "llm":
                        agent_prompt = load_markdown_file(user_prompt_path).format(
                            files=files_used_formatted_str_llm,
                            question=question,
                            packages=packages_used_bullet_str,
                        )

                    try:
                        eda_execution = sandbox_eda.execute_code(eda_code)

                    except Exception as e:
                        print(f"Error executing code: {e}")

                    if len(eda_execution) > 0:
                        # Add code cell with EDA execution results

                        notebook_data["cells"].append(
                            {
                                "cell_type": "markdown",
                                "metadata": {},
                                "source": assistant_thinking_template.format(
                                    full_response
                                )
                                .replace("<think>", "&lt;think&gt;")
                                .replace("</think>", "&lt;/think&gt;"),
                            }
                        )

                        if (
                            tool_call.function.name
                            == "add_and_execute_jupyter_code_cell"
                        ):
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": full_response,
                                    "tool_calls": [
                                        {
                                            "id": tool_call.id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_call.function.name,
                                                "arguments": tool_call.function.arguments,
                                            },
                                        }
                                    ],  # type: ignore
                                }
                            )

                            notebook_data["cells"].append(
                                {
                                    "cell_type": "code",
                                    "execution_count": None,
                                    "metadata": {},
                                    "source": eda_code,
                                    "outputs": sandbox_eda.parse_exec_result_nb(
                                        eda_execution
                                    ),
                                }
                            )

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": "1",
                                    "content": eda_execution,
                                }
                            )
                else:
                    eda_execution = "No information provided"

                if type_executor == "e2b":
                    # pass the initialized E2B sandbox to the main loop
                    sandbox = sandbox_eda
                else:
                    sandbox = Sandbox(
                        self.router_ip,
                        self.router_port,
                        original_notebook,
                        doc.metadata["files_used"],
                    )

                # Process the document through multiple turns
                document_processed = False

                for _ in range(self.max_turns):
                    tool_calls = None
                    if document_processed:
                        break

                    tool_no_call_count = 0

                    while tool_calls is None:
                        tool_no_call_count += 1
                        if tool_no_call_count > 5:
                            print(
                                f"Too many tool calls without arguments for document {doc.id}"
                            )
                            if type_executor == "e2b":
                                sandbox.sandbox.kill()
                            return None

                        try:
                            response = client.chat.completions.create(
                                messages=messages,
                                model=model_gen_name,
                                tools=tools,
                            )
                        except Exception as e:
                            print(f"Error in API call: {e} for document {doc.id}")
                            continue

                        # Get the response content and tool calls
                        full_response = response.choices[0].message.content or ""
                        tool_calls = response.choices[0].message.tool_calls or None
                        if tool_calls:
                            tool_args = json.loads(tool_calls[0].function.arguments)
                        else:
                            tool_calls = None

                        if "Coder" in model_gen_name:
                            if tool_calls:
                                if tool_args and "comment" in tool_args:
                                    full_response = (
                                        "<think>" + tool_args["comment"] + "</think>"
                                    )
                            elif (
                                response.choices[0].message.content
                                and len(response.choices[0].message.content) > 0
                            ):
                                full_response = (
                                    "<think>"
                                    + response.choices[0].message.content
                                    + "</think>"
                                )
                            else:
                                full_response = "<think>Executing next step...</think>"
                        else:
                            full_response = response.choices[0].message.content

                    # Add markdown cell for assistant's thinking
                    notebook_data["cells"].append(
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": assistant_thinking_template.format(full_response)
                            .replace("<think>", "&lt;think&gt;")
                            .replace("</think>", "&lt;/think&gt;"),
                        }
                    )

                    # Handle tool calls
                    for tool_call in tool_calls:
                        # Add assistant message with tool calls
                        messages.append(
                            {
                                "role": "assistant",
                                "content": full_response,
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments,
                                        },
                                    }
                                ],
                            }
                        )

                        if (
                            tool_call.function.name
                            == "add_and_execute_jupyter_code_cell"
                        ):
                            tool_args = json.loads(tool_call.function.arguments)

                            if "code" not in tool_args:
                                print(
                                    f"Error: code not in tool_args for document {doc.id}"
                                )
                                return None

                            tool_args["code"] = extract_code_from_block(
                                tool_args["code"]
                            )

                            try:
                                execution = sandbox.execute_code(tool_args["code"])
                            except Exception as e:
                                print(
                                    f"Error executing code for document {doc.id}: {e}"
                                )
                                return None

                            outputs = sandbox.parse_exec_result_nb(execution)

                            # Add code cell with execution results
                            notebook_data["cells"].append(
                                {
                                    "cell_type": "code",
                                    "execution_count": None,
                                    "metadata": {},
                                    "source": tool_args["code"],
                                    "outputs": outputs,
                                }
                            )

                            # Add tool response to messages
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": execution,
                                }
                            )

                        elif tool_call.function.name == "add_markdown_cell":
                            tool_args = json.loads(tool_call.function.arguments)
                            notebook_data["cells"].append(
                                {
                                    "cell_type": "markdown",
                                    "metadata": {},
                                    "source": tool_args["content"],
                                }
                            )

                        elif tool_call.function.name == "final_answer":
                            tool_args = json.loads(tool_call.function.arguments)
                            notebook_data["cells"].append(
                                {
                                    "cell_type": "markdown",
                                    "metadata": {},
                                    "source": assistant_final_answer_template.format(
                                        tool_args["answer"]
                                    ),
                                }
                            )

                            # Update document with the final notebook
                            doc.text = {
                                **doc_text_data,
                                "notebook": notebook_data,
                            }

                            doc.metadata["executor_type"] = type_executor

                            document_processed = True
                            print("Processed doc ", doc.id)
                            if type_executor == "e2b":
                                sandbox.sandbox.kill()
                            yield doc  # Return the processed document

                    # If no tool calls, break (this should not happen normally)
                    if not tool_calls:
                        print(f"Warning: No tool calls for document {doc.id}")
                        if type_executor == "e2b":
                            sandbox.sandbox.kill()
                        return None

                # If we reached max_turns without final_answer
                if not document_processed:
                    print(
                        f"Warning: Document {doc.id} reached max_turns without final_answer"
                    )
                    if type_executor == "e2b":
                        sandbox.sandbox.kill()
                    return None

            except Exception as e:
                print(f"Error processing document {getattr(doc, 'id', 'unknown')}: {e}")
                return None

            yield doc


def main():
    import argparse

    # Parse command line arguments
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

    data_folder = "/fsx/jupyter-agent/data/kaggle-mapped"
    output_folder = "/fsx/jupyter-agent/data/generate-traces"

    logs_file = "logs/qwen3-llm-traces"

    pipeline = [
        JsonlReader(data_folder=data_folder, limit=LIMIT),
        GenerateTraces(
            router_ip=router_ip, router_port=router_port
        ),
        DataAgentTracesNotebookToChatML(),
        JsonlWriter(output_folder=output_folder, max_file_size=int(10**9)),
    ]

    executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        tasks=N_TASKS,
        workers=N_WORKERS,
        time="48:00:00",
        partition="hopper-prod",
        logging_dir=logs_file,
        cpus_per_task=4,
        mem_per_cpu_gb=32,
        qos="high",
        skip_completed=False,
        job_name="generate-traces",
    )

    executor.run()


if __name__ == "__main__":
    main()
