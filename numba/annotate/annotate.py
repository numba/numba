# -*- coding: UTF-8 -*-

"""
numba --annotate
"""

from __future__ import print_function, division, absolute_import

import sys
import operator
from itertools import groupby, chain
from collections import namedtuple

# ______________________________________________________________________

Program = namedtuple("Program", ["python_source", "intermediates"])
SourceIntermediate = namedtuple("SourceIntermediate", ["name", "linenomap",
                                                       "source"])
DotIntermediate = namedtuple("DotIntermediate", ["name", "dotcode"]) # graphviz
Source = namedtuple("Source", ["linemap", "annotations"])
Annotation = namedtuple("Annotation", ["type", "value"])

# ______________________________________________________________________

def build_linemap(func):
    import inspect
    import textwrap
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    lines = source.split('\n')
    if lines[-1] == '':
        lines = lines[0:-1]
    
    lineno = func.func_code.co_firstlineno
    linemap = {}
    for line in lines:
        linemap[lineno] = line
        lineno += 1

    return linemap

# ______________________________________________________________________

# Annotation types

A_type      = "Types"
A_c_api     = "Python C API"
A_numpy     = "NumPy"
A_errcheck  = "Error check"
A_objcoerce = "Coercion"
A_pycall    = "Python call"
A_pyattr    = "Python attribute"

#------------------------------------------------------------------------
# Helpers
#------------------------------------------------------------------------

Emitter = namedtuple("Emitter", ["emit", "emitline"])

def render_intermediates(program, intermediate_names, capability='source'):
    """
    Render intermediate representations:

        >>> render_intermediates(my_program, ["llvm"], "source")
        [("llvm", { 1: [1, 2, 3] }, Source(...))]
    """
    intermediates = dict((i.name, i) for i in program.intermediates)
    for intermediate_name in intermediate_names:
        intermediate = intermediates[intermediate_name]
        if isinstance(intermediate, SourceIntermediate):
            yield intermediate

#------------------------------------------------------------------------
# Text Rendering
#------------------------------------------------------------------------

WIDTH = 40
ANNOT_SEP = "-"

def _gather_text_annotations(annotations):
    adict = groupdict(annotations, 'type')
    for category, annotations in adict.items():
        vals = u" ".join([str(a.value) for a in annotations])
        yield u"%s: %s" % (category, vals)

def _gather_text_intermediates(intermediates, lineno):
    for irname, linenomap, ir_source in intermediates:
        ir_linenos = linenomap.get(lineno, [])
        if not ir_linenos:
            continue
        yield irname.center(WIDTH, "_")
        for ir_lineno in ir_linenos:
            yield ir_source.linemap[ir_lineno]

groupdict = lambda xs, attr: dict(
    (k, list(v)) for k, v in groupby(xs, operator.attrgetter(attr)))

# ______________________________________________________________________

def render_text(program, emit=sys.stdout.write,
                intermediate_names=(), inline=True):
    """
    Render a Program as text.

    :param intermediate_names: [intermediate_name], e.g. ["llvm"]
    :param inline: whether to display intermediate code inline
    """
    indent = 8
    emitline = lambda indent, s: emit(u" " * indent + s + u"\n")
    emitter = Emitter(emit, emitline)
    if program.intermediates:
        irs = list(render_intermediates(program, intermediate_names))
    else:
        irs = None

    # Render main source
    render_source(program.python_source, emitter, indent,
        irs if inline and irs else [])

    if not inline and irs:
        # Render IRs seperately
        for irname, linenomap, ir_source in irs:
            emitter.emitline(0, irname.center(80, "="))
            render_source(ir_source, emitter, indent, [], linenomap)
        emitter.emitline(0, "=" * 80)

def render_source(source, emitter, indent, intermediates, linenomap=None):
    if linenomap:
        indent += 8
        headers = {}
        for py_lineno, ir_linenos in linenomap.items():
            for ir_lineno in ir_linenos:
                headers[ir_lineno] = u"%4d |  " % py_lineno

        header = lambda lineno: headers.get(lineno, u"     |  ")
    else:
        header = lambda lineno: u""

    _render_source(source, emitter, indent, intermediates, header)


def _render_source(source, emitter, indent, intermediates, header=None):
    for lineno in sorted(source.linemap.iterkeys()):
        if header:
            emitter.emit(header(lineno))
        emitter.emitline(0, u"%4d    %s" % (lineno, source.linemap[lineno]))

        annots = _gather_text_annotations(source.annotations.get(lineno, []))
        irs = _gather_text_intermediates(intermediates, lineno)
        lines = list(chain(annots, irs))
        if not lines:
            continue

        # Print out annotations
        linestart = indent + len(source.linemap[lineno]) - len(source.linemap[lineno].lstrip())
        emitter.emitline(linestart + 2, u"||".center(WIDTH, ANNOT_SEP))
        for line in lines:
            emitter.emitline(linestart + 2, line)
        emitter.emitline(linestart + 2, u"||".center(WIDTH, ANNOT_SEP))

# ______________________________________________________________________

def render_webpage(program, emit=sys.stdout.write,
                   intermediate_names=(), inline=True):
    raise NotImplementedError