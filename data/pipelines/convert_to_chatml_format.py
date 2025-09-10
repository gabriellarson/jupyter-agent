from typing import Iterable
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.data import Document


class DataAgentTracesNotebookToChatML(PipelineStep):
    """Convert notebook JSON from GenerateTracesFromQuestion to ChatML format"""

    name = "🔄 DataAgentTracesNotebookToChatML"

    def __init__(self):
        super().__init__()

    def extract_content_from_html_template(
        self, html_content: str, template_type: str
    ) -> str:
        """Extract content from HTML templates used in notebook cells"""
        import re

        if template_type == "user":
            # Extract from: <div class="alert alert-block alert-success"><b>User:</b> CONTENT</div>
            match = re.search(r"<b>User:</b>\s*(.*?)</div>", html_content, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Replace <br> with newlines and clean up HTML
                content = content.replace("<br>", "\n").strip()
                return content

        elif template_type == "assistant_thinking":
            # Extract from: <div class="alert alert-block alert-danger"><b>Assistant:</b> CONTENT</div>
            match = re.search(
                r"<b>Assistant:</b>\s*(.*?)</div>", html_content, re.DOTALL
            )
            if match:
                content = match.group(1).strip()
                # Extract content between &lt;think&gt; and &lt;/think&gt;
                if (
                    "&lt;think&gt;" in content
                ):  # qwen-coder output does not have thinking tags
                    think_match = re.search(
                        r"&lt;think&gt;(.*?)&lt;/think&gt;", content, re.DOTALL
                    )
                    if think_match:
                        return think_match.group(1).strip()
                else:
                    think_match = content

        elif template_type == "assistant_final":
            # Extract from: <div class="alert alert-block alert-warning"><b>Assistant:</b> Final answer: CONTENT</div>
            match = re.search(
                r"<b>Assistant:</b>\s*Final answer:\s*(.*?)</div>",
                html_content,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()

        return ""

    def is_header_cell(self, cell: dict) -> bool:
        """Check if cell is the header/logo cell that should be ignored"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "jupyter-agent.png" in source or "Jupyter Agent Logo" in source
        return False

    def is_user_cell(self, cell: dict) -> bool:
        """Check if cell contains user message"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "alert-block alert-success" in source and "<b>User:</b>" in source
        return False

    def is_system_cell(self, cell: dict) -> bool:
        """Check if cell contains user message"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "alert-block alert-info" in source and "<b>System:" in source
        return False

    def is_assistant_thinking_cell(self, cell: dict) -> bool:
        """Check if cell contains assistant thinking"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return (
                "alert-block alert-danger" in source and "<b>Assistant:</b>" in source
            )
        return False

    def is_assistant_final_cell(self, cell: dict) -> bool:
        """Check if cell contains final answer"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "alert-block alert-warning" in source and "Final answer:" in source
        return False

    def is_code_cell(self, cell: dict) -> bool:
        """Check if cell is a code cell"""
        return cell.get("cell_type") == "code"

    def extract_code_execution_output(self, cell: dict) -> str:
        """Extract output from code cell"""
        outputs = cell.get("outputs", [])
        if not outputs:
            return ""

        output_parts = []
        for output in outputs:
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                output_parts.append(text)
            elif output.get("output_type") == "execute_result":
                data = output.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    if isinstance(text, list):
                        text = "".join(text)
                    output_parts.append(text)
            elif output.get("output_type") == "error":
                # Handle error outputs
                traceback = output.get("traceback", [])
                if traceback:
                    output_parts.append("".join(traceback))

        return "".join(output_parts).strip()

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        import json

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_and_execute_jupyter_code_cell",
                    "description": "A Python code execution environment that runs code in a Jupyter notebook interface. This is stateful - variables and imports persist between executions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Use this tool to provide the final answer to the user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The final answer to the user's question.",
                            }
                        },
                        "required": ["answer"],
                    },
                },
            },
        ]

        for doc in data:
            with self.track_time():
                try:
                    # Parse notebook JSON

                    if doc.metadata["question"] is None:
                        continue

                    notebook_data: dict = doc.text["notebook"]  # type: ignore

                    cells = notebook_data.get("cells", [])

                    # Initialize ChatML messages
                    chatml_messages = []

                    # Process cells in order
                    i = 0
                    current_thinking = None
                    unknown_cells_found = False

                    while i < len(cells):
                        cell = cells[i]

                        # Skip header cells
                        if self.is_header_cell(cell):
                            i += 1
                            continue

                        # Extract user message
                        elif self.is_user_cell(cell):
                            user_content = self.extract_content_from_html_template(
                                cell.get("source", ""), "user"
                            )
                            if user_content:
                                chatml_messages.append(
                                    {"role": "user", "content": user_content}
                                )
                            i += 1

                        # Extract system message
                        elif self.is_system_cell(cell):
                            system_content = self.extract_content_from_html_template(
                                cell.get("source", ""), "system"
                            )
                            if system_content:
                                chatml_messages.append(
                                    {"role": "system", "content": system_content}
                                )
                            i += 1

                        # Extract assistant thinking
                        elif self.is_assistant_thinking_cell(cell):
                            assert current_thinking is None, (
                                "current_thinking should be None as otherwise we would have double thinking, "
                                "so there is a problem earlier.\n"
                                # f"The document text is:\n{json.dumps(doc.text, indent=2)}"
                            )
                            current_thinking = self.extract_content_from_html_template(
                                cell.get("source", ""), "assistant_thinking"
                            )
                            i += 1

                        # Process code cell with thinking
                        elif self.is_code_cell(cell):
                            code_source = cell.get("source", "")

                            # Create assistant message with thinking and tool call
                            assistant_content = ""
                            if current_thinking:
                                assistant_content = (
                                    f"<think>\n{current_thinking}\n</think>\n\n"
                                )

                            chatml_messages.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_content,
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "add_and_execute_jupyter_code_cell",
                                                "arguments": {"code": code_source},
                                            }
                                        }
                                    ],
                                }
                            )

                            # Add tool response with execution output
                            execution_output = self.extract_code_execution_output(cell)
                            chatml_messages.append(
                                {"role": "tool", "content": execution_output}
                            )

                            current_thinking = None  # Reset thinking after use
                            i += 1

                        # Process final answer
                        elif self.is_assistant_final_cell(cell):
                            final_answer = self.extract_content_from_html_template(
                                cell.get("source", ""), "assistant_final"
                            )

                            assistant_content = ""
                            if current_thinking:
                                assistant_content = (
                                    f"<think>\n{current_thinking}\n</think>\n\n"
                                )

                            chatml_messages.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_content,
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "final_answer",
                                                "arguments": {"answer": final_answer},
                                            }
                                        }
                                    ],
                                }
                            )

                            current_thinking = None  # Reset thinking after use
                            i += 1

                        else:
                            # Unknown cell type found
                            unknown_cells_found = True
                            print(
                                f"Unknown cell type found in notebook for document {doc.id}: {cell.get('cell_type')} for cell {cell}"
                            )
                            i += 1

                    # If unknown cells were found, don't yield and print error
                    if unknown_cells_found:
                        print(
                            f"Error: Unknown cell type found in notebook for document {doc.id}"
                        )
                        print("Full notebook content that could not be parsed:")
                        print(json.dumps(notebook_data, indent=2))
                        continue

                    # Validate that we have at least user and assistant messages
                    if len(chatml_messages) < 2:
                        print(f"Insufficient messages extracted for document {doc.id}")
                        continue

                    # Update document with ChatML format
                    doc.text = {  # type: ignore
                        **doc.text,  # type: ignore
                        "messages": chatml_messages,
                        "tools": tools,
                    }
                except Exception as e:
                    print(f"Error: {e} for document\n")
                    continue

            yield doc


class DataAgentTracesNotebookToNonThinkingChatML(PipelineStep):
    """Convert notebook JSON from GenerateTracesFromQuestion to ChatML format"""

    name = "🔄 DataAgentTracesNotebookToChatML"

    def __init__(self):
        super().__init__()

    def extract_content_from_html_template(
        self, html_content: str, template_type: str
    ) -> str:
        """Extract content from HTML templates used in notebook cells"""
        import re

        if template_type == "user":
            # Extract from: <div class="alert alert-block alert-success"><b>User:</b> CONTENT</div>
            match = re.search(r"<b>User:</b>\s*(.*?)</div>", html_content, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Replace <br> with newlines and clean up HTML
                content = content.replace("<br>", "\n").strip()
                return content

        elif template_type == "assistant_thinking":
            # Extract from: <div class="alert alert-block alert-danger"><b>Assistant:</b> CONTENT</div>
            match = re.search(
                r"<b>Assistant:</b>\s*(.*?)</div>", html_content, re.DOTALL
            )
            if match:
                content = match.group(1).strip()
                # Extract content between &lt;think&gt; and &lt;/think&gt;
                if (
                    "&lt;think&gt;" in content
                ):  # qwen-coder output does not have thinking tags
                    think_match = re.search(
                        r"&lt;think&gt;(.*?)&lt;/think&gt;", content, re.DOTALL
                    )
                    if think_match:
                        return think_match.group(1).strip()
                else:
                    think_match = content

        elif template_type == "assistant_final":
            # Extract from: <div class="alert alert-block alert-warning"><b>Assistant:</b> Final answer: CONTENT</div>
            match = re.search(
                r"<b>Assistant:</b>\s*Final answer:\s*(.*?)</div>",
                html_content,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()

        return ""

    def is_header_cell(self, cell: dict) -> bool:
        """Check if cell is the header/logo cell that should be ignored"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "jupyter-agent.png" in source or "Jupyter Agent Logo" in source
        return False

    def is_user_cell(self, cell: dict) -> bool:
        """Check if cell contains user message"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "alert-block alert-success" in source and "<b>User:</b>" in source
        return False

    def is_system_cell(self, cell: dict) -> bool:
        """Check if cell contains user message"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "alert-block alert-info" in source and "<b>System:" in source
        return False

    def is_assistant_thinking_cell(self, cell: dict) -> bool:
        """Check if cell contains assistant thinking"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return (
                "alert-block alert-danger" in source and "<b>Assistant:</b>" in source
            )
        return False

    def is_assistant_final_cell(self, cell: dict) -> bool:
        """Check if cell contains final answer"""
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            return "alert-block alert-warning" in source and "Final answer:" in source
        return False

    def is_code_cell(self, cell: dict) -> bool:
        """Check if cell is a code cell"""
        return cell.get("cell_type") == "code"

    def extract_code_execution_output(self, cell: dict) -> str:
        """Extract output from code cell"""
        outputs = cell.get("outputs", [])
        if not outputs:
            return ""

        output_parts = []
        for output in outputs:
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                output_parts.append(text)
            elif output.get("output_type") == "execute_result":
                data = output.get("data", {})
                if "text/plain" in data:
                    text = data["text/plain"]
                    if isinstance(text, list):
                        text = "".join(text)
                    output_parts.append(text)
            elif output.get("output_type") == "error":
                # Handle error outputs
                traceback = output.get("traceback", [])
                if traceback:
                    output_parts.append("".join(traceback))

        return "".join(output_parts).strip()

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        import json

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_and_execute_jupyter_code_cell",
                    "description": "A Python code execution environment that runs code in a Jupyter notebook interface. This is stateful - variables and imports persist between executions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Use this tool to provide the final answer to the user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The final answer to the user's question.",
                            }
                        },
                        "required": ["answer"],
                    },
                },
            },
        ]

        for doc in data:
            with self.track_time():
                try:
                    # Parse notebook JSON

                    # print(f"doc.text: {doc.text}")

                    if doc.metadata["question"] is None:
                        continue

                    notebook_data: dict = doc.text["notebook"]  # type: ignore

                    cells = notebook_data.get("cells", [])

                    # Initialize ChatML messages
                    chatml_messages = []

                    # Process cells in order
                    i = 0
                    current_thinking = None
                    unknown_cells_found = False

                    while i < len(cells):
                        cell = cells[i]

                        # Skip header cells
                        if self.is_header_cell(cell):
                            i += 1
                            continue

                        # Extract user message
                        elif self.is_user_cell(cell):
                            user_content = self.extract_content_from_html_template(
                                cell.get("source", ""), "user"
                            )
                            if user_content:
                                # Remove any think tags if they somehow appear in user content
                                user_content = user_content.replace(
                                    "<think>", ""
                                ).replace("</think>", "")
                                chatml_messages.append(
                                    {"role": "user", "content": user_content}
                                )
                            i += 1

                        # Extract system message
                        elif self.is_system_cell(cell):
                            system_content = self.extract_content_from_html_template(
                                cell.get("source", ""), "system"
                            )
                            if system_content:
                                # Remove any think tags if they somehow appear in system content
                                system_content = system_content.replace(
                                    "<think>", ""
                                ).replace("</think>", "")
                                chatml_messages.append(
                                    {"role": "system", "content": system_content}
                                )
                            i += 1

                        # Extract assistant thinking
                        elif self.is_assistant_thinking_cell(cell):
                            assert current_thinking is None, (
                                "current_thinking should be None as otherwise we would have double thinking, "
                                "so there is a problem earlier.\n"
                                # f"The document text is:\n{json.dumps(doc.text, indent=2)}"
                            )
                            current_thinking = self.extract_content_from_html_template(
                                cell.get("source", ""), "assistant_thinking"
                            )
                            # Make sure to remove any think tags that might have been preserved
                            if current_thinking:
                                current_thinking = current_thinking.replace(
                                    "<think>", ""
                                ).replace("</think>", "")
                            i += 1

                        # Process code cell with thinking
                        elif self.is_code_cell(cell):
                            code_source = cell.get("source", "")

                            # Create assistant message without thinking and with tool call
                            assistant_content = ""
                            if current_thinking:
                                assistant_content = f"{current_thinking.replace('<think>', '').replace('</think>', '')}\n"

                            # Ensure all think tags are removed
                            assistant_content = assistant_content.replace(
                                "<think>", ""
                            ).replace("</think>", "")

                            chatml_messages.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_content,
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "add_and_execute_jupyter_code_cell",
                                                "arguments": {"code": code_source},
                                            }
                                        }
                                    ],
                                }
                            )

                            # Add tool response with execution output
                            execution_output = self.extract_code_execution_output(cell)
                            # Remove any think tags from execution output
                            execution_output = execution_output.replace(
                                "<think>", ""
                            ).replace("</think>", "")
                            chatml_messages.append(
                                {"role": "tool", "content": execution_output}
                            )

                            current_thinking = None  # Reset thinking after use
                            i += 1

                        # Process final answer
                        elif self.is_assistant_final_cell(cell):
                            final_answer = self.extract_content_from_html_template(
                                cell.get("source", ""), "assistant_final"
                            )
                            # Remove any think tags from final answer
                            final_answer = final_answer.replace("<think>", "").replace(
                                "</think>", ""
                            )

                            assistant_content = ""
                            if current_thinking:
                                assistant_content = f"{current_thinking.replace('<think>', '').replace('</think>', '')}\n"

                            # Ensure all think tags are removed
                            assistant_content = assistant_content.replace(
                                "<think>", ""
                            ).replace("</think>", "")

                            chatml_messages.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_content,
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "final_answer",
                                                "arguments": {
                                                    "answer": final_answer.replace(
                                                        "<think>", ""
                                                    ).replace("</think>", "")
                                                },
                                            }
                                        }
                                    ],
                                }
                            )

                            current_thinking = None  # Reset thinking after use
                            i += 1

                        else:
                            # Unknown cell type found
                            unknown_cells_found = True
                            print(
                                f"Unknown cell type found in notebook for document {doc.id}: {cell.get('cell_type')} for cell {cell}"
                            )
                            i += 1

                    # If unknown cells were found, don't yield and print error
                    if unknown_cells_found:
                        print(
                            f"Error: Unknown cell type found in notebook for document {doc.id}"
                        )
                        print("Full notebook content that could not be parsed:")
                        print(json.dumps(notebook_data, indent=2))
                        continue

                    # Validate that we have at least user and assistant messages
                    if len(chatml_messages) < 2:
                        print(f"Insufficient messages extracted for document {doc.id}")
                        continue

                    # Final pass to remove any remaining think tags from all messages
                    for message in chatml_messages:
                        if "content" in message and message["content"]:
                            message["content"] = (
                                message["content"]
                                .replace("<think>", "")
                                .replace("</think>", "")
                            )

                    # Update document with ChatML format
                    doc.text = {  # type: ignore
                        **doc.text,  # type: ignore
                        "messages": chatml_messages,
                        "tools": tools,
                    }
                except Exception as e:
                    print(
                        f"Error: {e} for document\n"
                        # f"The document is:\n{json.dumps(doc.text, indent=2)}"
                    )
                    continue

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

    data_folder = "/fsx/jupyter-agent/data/generate-traces"
    output_folder = "/fsx/jupyter-agent/data/chatml-dataset-traces"

    logs_file = "logs/qwen-coder-traces-threaded"

    thinking = True

    if thinking:
        chatml_step = DataAgentTracesNotebookToChatML()
    else:
        chatml_step = DataAgentTracesNotebookToNonThinkingChatML()

    pipeline = [
        JsonlReader(data_folder=data_folder, limit=LIMIT),
        chatml_step,
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
    )

    executor.run()


if __name__ == "__main__":
    main()
