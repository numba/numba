# -*- coding: UTF-8 -*-

"""
HTML annotation rendering. Heavily based on Cython/Compiler/Annotate.py
"""

from __future__ import print_function, division, absolute_import

import sys
import cgi
import os
import re
from .annotate import format_annotations, groupdict, A_c_api
from .step import Template

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.lexers import LlvmLexer
    from pygments.formatters import HtmlFormatter
    pygments_installed = True
except ImportError:
    pygments_installed = False

def render(annotation_blocks, emit=sys.stdout.write,
           intermediate_names=(), inline=True):
    """
    Render a Program as html.
    """
    root = os.path.join(os.path.dirname(__file__))
    if inline:
        templatefile = os.path.join(root, 'annotate_inline_template.html')
    else:
        templatefile = os.path.join(root, 'annotate_template.html')

    with open(templatefile, 'r') as f:
        template = f.read()

    py_c_api = re.compile(u'(Py[A-Z][a-z]+_[A-Z][a-z][A-Za-z_]+)\(')

    data = {'blocks': []}

    for i, block in enumerate(annotation_blocks):
        python_source = block['python_source']
        intermediates = block['intermediates']
        data['blocks'].append({'lines':[]})

        for num, source in sorted(python_source.linemap.items()):

            types = {}
            if num in python_source.annotations.keys():
                for a in python_source.annotations[num]:
                    if a.type == 'Types':
                        name = a.value[0]
                        type = a.value[1]
                        types[name] = type
            
            types_str = ','.join(name + ':' + type for name, type in types.items())

            python_calls = 0
            llvm_nums = intermediates[0].linenomap[num]
            llvm_ir = ''
            for llvm_num in llvm_nums:
                ir = intermediates[0].source.linemap[llvm_num]
                if re.search(py_c_api, ir):
                    python_calls += 1

                if pygments_installed:

                    class LlvmHtmlFormatter(HtmlFormatter):
                        def wrap(self, source, outfile):
                            return self._wrap_code(source)
                        def _wrap_code(self, source):
                            for i, t in source:
                                yield i, t

                    ir = highlight(ir, LlvmLexer(), LlvmHtmlFormatter())
                       
                llvm_ir += '<div>' + ir + '</div>'

            if python_calls > 0:
                tag = '*'
                tag_css = 'tag'
            else:
                tag = ''
                tag_css = ''

            if num == python_source.linemap.keys()[0]:
                firstlastline = 'firstline'
            elif num == python_source.linemap.keys()[-1]:
                firstlastline = 'lastline'
            else:
                firstlastline = 'innerline'

            if pygments_installed:

                class PythonHtmlFormatter(HtmlFormatter):
                    def wrap(self, source, outfile):
                        return self._wrap_code(source)
                    def _wrap_code(self, source):
                        for i, t in source:
                            yield i, t

                source = highlight(source, PythonLexer(), PythonHtmlFormatter())
               
            data['blocks'][-1]['func_call'] = block['func_call']
            data['blocks'][-1]['func_call_filename'] = block['func_call_filename']
            data['blocks'][-1]['func_call_lineno'] = block['func_call_lineno']
            data['blocks'][-1]['lines'].append({'id': str(i) + str(num),
                                  'num':str(num) + tag,
                                  'tag':tag_css,
                                  'python_source':source,
                                  'llvm_source':llvm_ir,
                                  'types':types_str,
                                  'firstlastline':firstlastline})

    css_theme_file = os.path.join(root, 'jquery-ui.min.css')
    with open(css_theme_file, 'r') as f:
        css_theme = f.read()
    data['jquery_theme'] = css_theme

    jquery_lib_file = os.path.join(root, 'jquery.min.js')
    with open(jquery_lib_file, 'r') as f:
        jquery_lib = f.read()
    data['jquery_lib'] = jquery_lib

    jquery_ui_lib_file = os.path.join(root, 'jquery-ui.min.js')
    with open(jquery_ui_lib_file, 'r') as f:
        jquery_ui_lib = f.read()
    data['jquery_ui_lib'] = jquery_ui_lib

    html = Template(template).expand(data)

    emit(html)
