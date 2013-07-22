# -*- coding: UTF-8 -*-
from __future__ import print_function, division, absolute_import

from io import StringIO
from numba.annotate.annotate import (Source, Annotation, Intermediate, Program,
                                     A_type, render_text, Renderer)

# ______________________________________________________________________

py_source = Source(
    linemap={ 1: u'def foo(a, b):',
              2: u'    print a * b',
              3: u'    a / b',
              4: u'    return a - b' },
    annotations={ 2: [Annotation(A_type, (u'a', u'double')),
                      Annotation(A_type, (u'b', u'double'))] }
)

class LLVMRenderer(Renderer):
    capabilities = frozenset(["source"])

    def render(self, capability):
        linenomap = { 1: [0], 2: [1, 2, 3], 4: [5], }
        llvm_linemap = {
            0: u'call @printf(%a, %b)',
            1: u'%0 = load a',
            2: u'%1 = load b',
            3: u'%2 = fadd %0 %1',
            4: u'%3 = fdiv %a %b',
            5: u'ret something',
        }
        annotations = {
            3: [Annotation(A_type, (u'%0', u'double')),
                Annotation(A_type, (u'%1', u'double'))],
        }

        return linenomap, Source(llvm_linemap, annotations)


llvm_intermediate = Intermediate("llvm", LLVMRenderer())
p = Program(py_source, [llvm_intermediate])

# ______________________________________________________________________

def run_render_text():
    f = StringIO()
    render_text(p, emit=f.write)
    src = f.getvalue()
    assert 'def foo(a, b):' in src
    assert 'print a * b' in src
    assert 'return a - b' in src
    assert 'double' in src
    # print(src)

def run_render_text_inline():
    f = StringIO()
    render_text(p, emit=f.write, intermediate_names=["llvm"])
    src = f.getvalue()
    assert 'def foo(a, b):' in src
    assert '____llvm____' in src
    assert '%0 = load a' in src
    # print(src)

def run_render_text_outline():
    f = StringIO()
    render_text(p, emit=f.write, inline=False, intermediate_names=["llvm"])
    src = f.getvalue()
    assert 'def foo(a, b):' in src
    assert "====llvm====" in src
    assert '%0 = load a' in src
    # print(src)

run_render_text()
run_render_text_inline()
run_render_text_outline()
