system_template = """\
<details>
  <summary style="display: flex; align-items: center;">
    <div class="alert alert-block alert-info" style="margin: 0; width: 100%;">
      <b>System: <span class="arrow">▶</span></b>
    </div>
  </summary>
  <div class="alert alert-block alert-info">
    {}
  </div>
</details>

<style>
details > summary .arrow {{
  display: inline-block;
  transition: transform 0.2s;
}}
details[open] > summary .arrow {{
  transform: rotate(90deg);
}}
</style>
"""

user_template = """<div class="alert alert-block alert-success">
<b>User:</b> {}
</div>
"""

assistant_thinking_template = """<div class="alert alert-block alert-danger">
<b>Assistant:</b> {}
</div>
"""

assistant_final_answer_template = """<div class="alert alert-block alert-warning">
<b>Assistant:</b> Final answer: {}
</div>
"""

header_message = """<p align="center">
  <img src="https://huggingface.co/spaces/lvwerra/jupyter-agent/resolve/main/jupyter-agent.png" alt="Jupyter Agent Logo" />
</p>


<p style="text-align:center;">Let a LLM agent write and execute code inside a notebook!</p>"""

bad_html_bad = """input[type="file"] {
  display: block;
}"""


def create_base_notebook(messages):
    base_notebook = {
        "metadata": {
            "kernel_info": {"name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
        "cells": [],
    }
    base_notebook["cells"].append(
        {"cell_type": "markdown", "metadata": {}, "source": header_message}
    )

    if len(messages) == 0:
        base_notebook["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": "",
                "outputs": [],
            }
        )

    code_cell_counter = 0

    for message in messages:
        if message["role"] == "system":
            text = system_template.format(message["content"].replace("\n", "<br>"))
            base_notebook["cells"].append(
                {"cell_type": "markdown", "metadata": {}, "source": text}
            )
        elif message["role"] == "user":
            text = user_template.format(message["content"].replace("\n", "<br>"))
            base_notebook["cells"].append(
                {"cell_type": "markdown", "metadata": {}, "source": text}
            )

        elif message["role"] == "assistant" and "tool_calls" in message:
            base_notebook["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": message["content"],
                    "outputs": [],
                }
            )

        elif message["role"] == "assistant" and "tool_calls" not in message:
            base_notebook["cells"].append(
                {"cell_type": "markdown", "metadata": {}, "source": message["content"]}
            )

        else:
            raise ValueError(message)

    return base_notebook, code_cell_counter


def extract_code_from_block(code_text):
    # Handle python/py code blocks
    if code_text.startswith("```python"):
        return code_text.split("```python")[1].split("```")[0].strip()
    elif code_text.startswith("```py"):
        return code_text.split("```py")[1].split("```")[0].strip()
    elif code_text.startswith("```"):
        # Handle generic code blocks
        return code_text.split("```")[1].split("```")[0].strip()
    else:
        return code_text  # No code block formatting


def parse_exec_result_e2b(execution):
    output = []
    trunc_limit_output = 25
    trunc_limit_error = 10

    def truncate_lines(text, trunc_limit):
        lines = text.splitlines()
        if len(lines) > trunc_limit:
            return (
                "\n".join(lines[:trunc_limit])
                + f"\n[Output is truncated as it is more than {trunc_limit} lines]"
            )
        else:
            return text

    def truncate_lines_inverse(text, trunc_limit):
        lines = text.splitlines()
        if len(lines) > trunc_limit:
            return (
                "\n".join(lines[:trunc_limit])
                + f"\n[Output is truncated to last few lines as it is more than {trunc_limit} lines]"
            )
        else:
            return text

    # Handle results
    for result in execution.results:
        if result.text:
            output.append(truncate_lines(result.text, trunc_limit_output))

    # Handle stdout
    if len(execution.logs.stdout) > 0:
        std_str = "\n".join(execution.logs.stdout)
        output.append(truncate_lines(std_str, trunc_limit_error))

    # Handle stderr
    if len(execution.logs.stderr) > 0:
        std_str = "\n".join(execution.logs.stderr)
        output.append(truncate_lines(std_str, trunc_limit_error))

    # Handle error traceback
    if execution.error is not None:
        std_str = execution.error.traceback
        output.append(truncate_lines_inverse(std_str, trunc_limit_error))

    return "\n".join(output)
