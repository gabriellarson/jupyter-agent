from openai import OpenAI
from .qwen3_formatter_utils import (
    load_markdown_file,
    parse_think_and_answer,
    parse_tagged_content,
)


class Sandbox:
    """
    This is a LLM based sandbox that can execute code in a Jupyter notebook environment.
    It uses a code_prompt to make the LLM act as a code interpreter based on a executed jupyter notebook.
    """

    def __init__(
        self,
        router_ip: str,
        router_port: int,
        markdown_notebook: str,
        file_list: list[str],
    ):
        self.router_ip = router_ip
        self.router_port = router_port
        self.markdown_notebook = markdown_notebook
        self.file_list_str = "\n".join(f"- {file}" for file in file_list)

        self.client = OpenAI(
            api_key="token-abc123",
            base_url=f"http://{self.router_ip}:{self.router_port}/v1",
        )

        self.system_prompt = load_markdown_file(
            "./pipelines/prompts/code_interpreter_system_prompt.md"
        )
        self.user_prompt = load_markdown_file(
            "./pipelines/prompts/code_interpreter_user_prompt.md"
        )

        self.code_execution_state = []

    def parse_exec_result_nb(self, execution: str):
        return [{"output_type": "stream", "name": "stdout", "text": execution}]

    def _format_execution_history(self):
        """Format the execution history into the proper notebook cell format"""
        if not self.code_execution_state:
            return "No previous execution history."

        formatted_history = []
        for i, entry in enumerate(self.code_execution_state):
            # Python cell
            formatted_history.append("<python_cell>")
            formatted_history.append(f"<id:cell_{i + 1}>")
            formatted_history.append(entry["code"])
            formatted_history.append("</python_cell>")
            formatted_history.append("")

            # Output cell
            formatted_history.append("<output_cell>")
            formatted_history.append(f"<id:cell_{i + 1}>")
            formatted_history.append(entry["execution"])
            formatted_history.append("</output_cell>")
            formatted_history.append("")

        return "\n".join(formatted_history)

    def execute_code(self, code):
        # Format execution history properly
        execution_history = self._format_execution_history()

        # Use the user prompt template with proper placeholders
        user_prompt = self.user_prompt.format(
            code=code,
            execution_history=execution_history,
            markdown_notebook=self.markdown_notebook,
            available_files_list=self.file_list_str,
        )

        model_gen_name = "Qwen3-Coder-480B-A35B-Instruct"

        response = self.client.chat.completions.create(
            model=model_gen_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },  # Use formatted user prompt, not raw code
            ],
        )

        # Remove the thinking part
        response_content = response.choices[0].message.content
        if "Coder" in model_gen_name:
            response_content = "<think>placeholder</think>" + response_content

        assert response_content is not None, "response_content is None"

        # this can raise an error if the response_content is not a valid response
        think, answer = parse_think_and_answer(response_content)

        output_cell = parse_tagged_content(answer)["output_cell"]

        # Keep only the first 25 lines of the answer if truncated
        lines = output_cell.split("\n")
        if len(lines) > 25:
            execution = "\n".join(lines[:25])
            execution += "\n[Output is truncated as it is more than 25 lines]"
        else:
            execution = output_cell

        print("##### LLM OUTPUT #####")
        print(execution)
        print("######################\n")

        # Store the execution state
        self.code_execution_state.append(
            {
                "code": code,
                "execution": execution,
            }
        )

        return execution


class SandboxE2B:
    def __init__(self, markdown_notebook: str, sandbox: Sandbox, doc):
        self.sandbox = sandbox
        self.markdown_notebook = markdown_notebook
        self.doc = doc

        self.code_execution_state = []

    def parse_exec_result_nb(self, execution: str):
        return [{"output_type": "stream", "name": "stdout", "text": execution}]

    def format_e2b_result(self, result):
        output = ""

        if result.text:
            output += str(result.text) + "\n"
        if result.latex:
            output += str(result.latex) + "\n"
        if result.json:
            output += str(result.json) + "\n"

        return output

    def _format_execution_history(self):
        """Format the execution history into the proper notebook cell format"""

        # run through the cells before the cell of interest? or just reuse old sandbox on notebook level

        if not self.code_execution_state:
            return "No previous execution history."

        for i, entry in enumerate(self.code_execution_state):
            self.sandbox.run_code(entry["code"])

    def execute_code(self, code):
        from .jupyter_utils import parse_exec_result_e2b

        code = code.replace("../input/", "/home/user/input/")
        code = code.replace("../kaggle/input/", "/home/user/input/")

        result = ""
        if code:
            execution = self.sandbox.run_code(str(code))
            result = parse_exec_result_e2b(execution)
        else:
            result = "No trace provided"

        print("##### E2B OUTPUT #####")
        print(result)
        print("######################\n")

        self.code_execution_state.append(
            {
                "code": code,
                "execution": result,
            }
        )

        return result
