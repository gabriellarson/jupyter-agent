from typing import List, Optional
import re
from typing import Any


tools_non_thinking: Any = [
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
                    },
                    "comment": {
                        "type": "string",
                        "description": "A detailed explanation of what is your logic when generating solution, what the code does, describing the task execution plan, be verbose - *required* minimum 5 sentences, ideally 10. For data analysis code use minimum 10 sentences.",
                    },
                },
                "required": ["code", "comment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer to the user's question after completing all necessary analysis and computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete final answer to the user's question - should be short and concise.",
                    },
                    "comment": {
                        "type": "string",
                        "description": "A detailed explanation of what is your logic when generating the final answer, be verbose - *required* minimum 5 sentences, ideally 10.'",
                    },
                },
                "required": ["answer", "comment"],
            },
        },
    },
]

tools_thinking: Any = [
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
            "description": "Provide the final answer to the user's question after completing all necessary analysis and computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete final answer to the user's question - should be short and concise.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


def clean_llm_response(text):
    """Remove <think> tags and their content from LLM response"""
    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned.strip()


def load_markdown_file(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()


def extract_original_notebook_content(doc_text, skip_markdown_cells: bool = False):
    """Extract the original notebook content (cells without Q&A metadata)"""

    # Find all cell blocks
    cell_pattern = (
        r"<(python|output|markdown)_cell>.*?</\1_cell>"
        if not skip_markdown_cells
        else r"<(python|output)_cell>.*?</\1_cell>"
    )
    cells = re.findall(cell_pattern, doc_text, re.DOTALL)

    # Join cells back together as original notebook
    return "\n\n".join(cells)


def extract_qa_pairs(text: str) -> List[tuple]:
    """Extract all question-answer pairs from text

    Returns:
        List of tuples: (question_num, question, answer, max_cell_id)
    """
    # FIXED: Use correct closing tags - was looking for </start_question_1> instead of </end_question_1>
    qa_pattern = r"<question_(\d+)>(.*?)</question_\1>\s*<answer_\1>(.*?)</answer_\1>"

    qa_pairs = []
    for match in re.finditer(qa_pattern, text, re.DOTALL):
        question_num = match.group(1)
        question = match.group(2).strip()
        answer = match.group(3).strip()

        # Extract cell ID from question if present
        cell_id_match = re.search(r"<id:cell_(\d+)>", question)
        max_cell_id = int(cell_id_match.group(1)) if cell_id_match else None

        # Remove the cell ID tag from the question text
        question = re.sub(r"<id:cell_\d+>\n?", "", question)

        qa_pairs.append((question_num, question, answer, max_cell_id))

    return qa_pairs


def has_qa_pairs(text: str) -> bool:
    """Check if text contains at least one Q&A pair"""
    return len(extract_qa_pairs(text)) > 0


def extract_tagged_content(text: str, tag_name: str) -> Optional[str]:
    """Extract content from a specific tag

    Args:
        text: Input text containing tagged content
        tag_name: Name of the tag to extract (without start_/end_ prefixes)

    Returns:
        Extracted content or None if tag not found
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def parse_think_and_answer(response_text: str) -> tuple[str, str]:
    """Parse the LLM response to extract thinking and answer sections from reasoning model"""

    # Extract thinking section from <think> tags
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Extract answer as everything outside <think> tags
    # Remove <think>...</think> sections to get the answer
    answer = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    answer = answer.strip()

    if not thinking:
        raise ValueError(
            f"Could not extract thinking section from response is '{response_text}'"
        )
    if not answer:
        raise ValueError(
            f"Could not extract answer section from response is '{response_text}'"
        )

    return thinking, answer


def parse_tagged_content(text):
    """Parse text containing tagged content and return the full content with tags"""
    # Pattern to match any tag and its content
    pattern = r"<(.*?)>(.*?)</\1>"
    matches = re.finditer(pattern, text, re.DOTALL)

    parsed_content = {}
    for match in matches:
        tag_name = match.group(1)
        content = match.group(2).strip()
        parsed_content[tag_name] = content

    return parsed_content
