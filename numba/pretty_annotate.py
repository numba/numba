"""
This module implements code highlighting of numba function annotations.

Example:

    >>> import numba 
    ... from numba_annotate import Annotate
    ... @numba.jit
    ... def test(q):
    ...     res = 0
    ...     for i in range(q):
    ...         res += i
    ...     return res
    ...     
    ... test(10)
    ... Annotate(test)

The last line will return an HTML and/or ANSI representation that will be
displayed accordingly in IPython/Jupyter. 

"""

from warnings import warn

warn("The pretty_annotate functionality is experimental and might change API",
         FutureWarning)

from jinja2 import Template
from pygments import highlight

from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
# TerminalFormatter does not support themes, Terminal256 should, but seem to not work.
from pygments.formatters import TerminalFormatter

pylex = PythonLexer()

def hllines(code, style):
    "Given a code string, return a list of html-highlighted lines"
    hf = HtmlFormatter(noclasses=True, style=style, nowrap=True)
    res = highlight(code, pylex, hf)
    return res.splitlines()


def htlines(code, style):
    "Given a code string, return a list of ANSI-highlighted lines"
    hf = TerminalFormatter(style=style)
    res = highlight(code, pylex, hf)
    return res.splitlines()

ansi_template = Template("""
{%- for func_key in func_data.keys() -%}
    Function name: \x1b[34m{{func_data[func_key]['funcname']}}\x1b[39;49;00m
    {%- if func_data[func_key]['filename'] -%}
    {{'\n'}}In file: \x1b[34m{{func_data[func_key]['filename'] -}}\x1b[39;49;00m
    {%- endif -%}
    {{'\n'}}With signature: \x1b[34m{{func_key[1]}}\x1b[39;49;00m
    {{- "\n" -}}
    {%- for num, line, hl, hc in func_data[func_key]['pygments_lines'] -%}
            {{-'\n'}}{{ num}}: {{hc-}}
            {%- if func_data[func_key]['ir_lines'][num] -%}
                {%- for ir_line, ir_line_type in func_data[func_key]['ir_lines'][num] %}
                    {{-'\n'}}--{{- ' '*func_data[func_key]['python_indent'][num]}}
                    {{- ' '*(func_data[func_key]['ir_indent'][num][loop.index0]+4)
                    }}{{ir_line }}\x1b[41m{{ir_line_type-}}\x1b[39;49;00m
                {%- endfor -%}
            {%- endif -%}
        {%- endfor -%}
{%- endfor -%}
""")

html_template = Template("""
<html>
<head>
    <style>
        .annotation_table {
            color: #000000;
            font-family: monospace;
            margin: 5px;
            width: 100%;
        }

        /* override JupyterLab style */
        .annotation_table td {
            text-align: left;
            background-color: transparent; 
            padding: 1px;
        }

        .annotation_table tbody tr:nth-child(even) {
            background: white;
        }

        .annotation_table code
        {
            background-color: transparent; 
            white-space: normal;
        }

        /* End override JupyterLab style */

        tr:hover {
            background-color: rgba(92, 200, 249, 0.25);
        }

        td.object_tag summary ,
        td.lifted_tag summary{
            font-weight: bold;
            display: list-item;
        }

        span.lifted_tag {
            color: #00cc33;
        }

        span.object_tag {
            color: #cc3300;
        }


        td.lifted_tag {
            background-color: #cdf7d8;
        }

        td.object_tag {
            background-color: #fef5c8;
        }

        code.ir_code {
            color: grey;
            font-style: italic;
        }

        .metadata {
            border-bottom: medium solid black;
            display: inline-block;
            padding: 5px;
            width: 100%;
        }

        .annotations {
            padding: 5px;
        }

        .hidden {
            display: none;
        }

        .buttons {
            padding: 10px;
            cursor: pointer;
        }
    </style>
</head>

<body>
    {% for func_key in func_data.keys() %}
        <div class="metadata">
        Function name: {{func_data[func_key]['funcname']}}<br />
        {% if func_data[func_key]['filename'] %}
            in file: {{func_data[func_key]['filename']|escape}}<br />
        {% endif %}
        with signature: {{func_key[1]|e}}
        </div>
        <div class="annotations">
        <table class="annotation_table tex2jax_ignore">
            {%- for num, line, hl, hc in func_data[func_key]['pygments_lines'] -%}
                {%- if func_data[func_key]['ir_lines'][num] %}
                    <tr><td class="{{func_data[func_key]['python_tags'][num]}}">
                        <details>
                            <summary>
                                <code>
                                {{num}}:
                                {{'&nbsp;'*func_data[func_key]['python_indent'][num]}}{{hl}}
                                </code>
                            </summary>
                            <table class="annotation_table">
                                <tbody>
                                    {%- for ir_line, ir_line_type in func_data[func_key]['ir_lines'][num] %}
                                        <tr class="ir_code">
                                            <td><code>
                                            &nbsp;
                                            {{- '&nbsp;'*func_data[func_key]['python_indent'][num]}}
                                            {{ '&nbsp;'*func_data[func_key]['ir_indent'][num][loop.index0]}}{{ir_line|e -}}
                                            <span class="object_tag">{{ir_line_type}}</span>
                                            </code>
                                            </td>
                                        </tr>
                                    {%- endfor -%}
                                </tbody>
                            </table>
                            </details>
                    </td></tr>
                {% else -%}
                    <tr><td style=" padding-left: 22px;" class="{{func_data[func_key]['python_tags'][num]}}">
                        <code>
                            {{num}}:
                            {{'&nbsp;'*func_data[func_key]['python_indent'][num]}}{{hl}}
                        </code>
                    </td></tr>
                {%- endif -%}
            {%- endfor -%}
        </table>
        </div>
    {% endfor %}
</body>
</html>
""")


def reform_code(annotation):
    """
    Extract the code from the Numba annotation datastructure. 

    Pygments can only highlight full multi-line strings, the Numba
    annotation is list of single lines, with indentation removed.
    """
    ident_dict = annotation['python_indent']
    s= ''
    for n,l in annotation['python_lines']:
        s = s+' '*ident_dict[n]+l+'\n'
    return s


class Annotate:
    """
    Construct syntax highlighted annotation for a given jitted function:

    Example:

    >>> import numba 
    ... from numba_annotate import Annotate
    ... @numba.jit
    ... def test(q):
    ...     res = 0
    ...     for i in range(q):
    ...         res += i
    ...     return res
    ...     
    ... test(10)
    ... Annotate(test)

    Function annotations persist across compilation for newly encountered
    type signatures and as a result annotations are shown for all signatures.

    """
    def __init__(self, function, **kwargs):

        style = kwargs.get('style', 'default')
        if not function.signatures:
            raise ValueError('function need to be jitted for at least one signature')
        for sig in function.signatures:
            ann = function.get_annotation_info(sig)
        self.ann = ann

        for k,v in ann.items():
            res = hllines(reform_code(v), style)
            rest = htlines(reform_code(v), style)
            v['pygments_lines'] = [(a,b,c, d) for (a,b),c, d in zip(v['python_lines'], res, rest)]

    def _repr_html_(self):
        return html_template.render(func_data=self.ann)

    def __repr__(self):
        return ansi_template.render(func_data=self.ann)
