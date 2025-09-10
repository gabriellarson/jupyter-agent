from datatrove.pipeline.base import PipelineStep
from datatrove.data import Document
from typing import Iterable
import os
import re
from jinja2 import FileSystemLoader
from nbconvert import TemplateExporter
from nbformat import NotebookNode
from traitlets.config import Config


class AddMarkdownToNotebook(PipelineStep):
    name = "🔄 AddMarkdownToNotebook"

    def __init__(self):
        super().__init__()
        self.template_file = os.path.abspath(
            "./pipelines/add_markdown_to_notebook/jupyter_to_markdown_template.tpl"
        )

    def convert_jupyter_to_markdown(self, notebook_text: str) -> str:
        """
        Convert a Jupyter notebook to custom markdown format using a Jinja template.

        Args:
            notebook_text (str): The notebook content as a JSON string

        Returns:
            str: The converted markdown string
        """
        import json
        import nbformat

        # Validate template file exists
        if not os.path.exists(self.template_file):
            raise FileNotFoundError(
                f"Template file does not exist: {self.template_file}"
            )

        # Initialize the template exporter
        exporter = TemplateExporter()

        # Set up the template loader with custom filter for image replacement
        template_dir = os.path.dirname(self.template_file)
        exporter.environment.loader = FileSystemLoader(template_dir)

        # Add minimal regex filter for image replacement
        exporter.environment.filters["regex_replace"] = (
            lambda text, pattern, replacement: re.sub(
                pattern, replacement, text, flags=re.DOTALL
            )
        )

        exporter.template_file = os.path.basename(self.template_file)

        # Configure the exporter
        config = Config()
        config.TemplateExporter.preprocessors = [
            "nbconvert.preprocessors.ExtractOutputPreprocessor"
        ]
        exporter.config = config

        # Parse the notebook from JSON string
        notebook_dict = json.loads(notebook_text)
        notebook = nbformat.from_dict(notebook_dict)

        assert isinstance(notebook, NotebookNode), "notebook is not a NotebookNode"

        # Convert the notebook using the template
        (output, resources) = exporter.from_notebook_node(notebook)

        # Process the output to remove empty cells and clean content
        processed_output = self.clean_markdown_output(output)

        return processed_output

    def clean_markdown_output(self, output: str) -> str:
        """
        Clean the markdown output by removing empty cells and normalizing whitespace.

        Args:
            output (str): Raw markdown output from the template

        Returns:
            str: Cleaned markdown output
        """
        # Pre-compile regex patterns for better performance
        cell_pattern = re.compile(
            r"(<(python|output|markdown)_cell>)(.*?)(</\2_cell>)", re.DOTALL
        )
        multiple_newlines = re.compile(r"\n{2,}")
        empty_lines = re.compile(r"\n\s*\n")

        def process_cell(match):
            start_tag = match.group(1)
            _cell_type = match.group(2)
            content = match.group(3)
            end_tag = match.group(4)

            # Strip content and remove empty lines
            content = content.strip()
            if not content:  # If cell is empty, return empty string
                return ""

            # Remove multiple empty lines and ensure single newlines
            content = empty_lines.sub("\n", content)

            # Return cell with newlines before and after
            return f"{start_tag}\n{content}\n{end_tag}"

        # Process all cells
        processed_output = cell_pattern.sub(process_cell, output)

        # Clean up any multiple consecutive newlines
        processed_output = multiple_newlines.sub("\n", processed_output)

        # Remove trailing newline and ensure no trailing whitespace
        processed_output = processed_output.strip()

        return processed_output

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterable[Document]:
        for doc in data:
            with self.track_time():
                try:
                    markdown = self.convert_jupyter_to_markdown(doc.text)
                    doc.text = {"markdown": markdown, "notebook": doc.text}  # type: ignore
                except Exception as e:
                    print(f"Error converting notebook to markdown: {e}")
                    continue

            yield doc
