{% set title = "Version {} (Release Date)".format(versiondata.version) %}

{{ title }}
{{ "-" * title|length }}

{% for section, _ in sections.items() %}
{% if section %}{{ section }}
{{ "~" * section|length }}

{% endif %}
{% if sections[section] %}
{% for category, val in definitions.items() if category in sections[section] %}

{{ definitions[category]['name'] }}
{{ "~" * definitions[category]['name']|length }}

{% if definitions[category]['showcontent'] %}
{% for text, values in sections[section][category].items() %}
{{ text }}

{{ get_indent(text) }}({{values|join(', ') }})

{% endfor %}
{% else %}
- {{ sections[section][category]['']|join(', ') }}

{% endif %}
{% if sections[section][category]|length == 0 %}
No significant changes.

{% else %}
{% endif %}
{% endfor %}
{% else %}
No significant changes.


{% endif %}
{% endfor %}