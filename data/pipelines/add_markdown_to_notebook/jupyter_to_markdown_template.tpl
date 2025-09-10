{%- block body -%}
{% for cell in nb.cells %}
{% if cell.cell_type == 'markdown' -%}
<markdown_cell>
<id:cell_{{ loop.index }}>
{%- set cleaned_source = cell.source | join('') | regex_replace('data:image/[^)"\s]+', '[IMAGE_PLACEHOLDER]') -%}
{{ cleaned_source }}
</markdown_cell>
{% elif cell.cell_type == 'code' -%}
<python_cell>
<id:cell_{{ loop.index }}>
{{ cell.source | join('') }}
</python_cell>

{% if cell.outputs -%}
<output_cell>
<id:cell_{{ loop.index }}>
{% for output in cell.outputs -%}
{% if output.output_type == 'stream' -%}
{% set full_text = output.text | join('') %}
{% if full_text | length > 5000 %}
{{ full_text[:5000] }}
[Output truncated at 5000 characters]
{% else %}
{{ full_text }}
{% endif %}
{% elif output.output_type == 'execute_result' -%}
{% if 'text/plain' in output.data -%}
{% set full_text = output.data['text/plain'] | join('') %}
{% if full_text | length > 5000 %}
{{ full_text[:5000] }}
[Output truncated at 5000 characters]
{% else %}
{{ full_text }}
{% endif %}
{% endif -%}
{% elif output.output_type == 'display_data' and 'text/plain' in output.data -%}
{% set full_text = output.data['text/plain'] | join('') %}
{% if full_text | length > 5000 %}
{{ full_text[:5000] }}
[Output truncated at 5000 characters]
{% else %}
{{ full_text }}
{% endif %}
{% elif output.output_type == 'error' -%}
{{ '\n'.join(output.traceback) }}
{% endif -%}
{% endfor -%}
</output_cell>
{% endif -%}
{% endif -%}
{% endfor -%}
{%- endblock body -%}