# -*- coding: UTF-8 -*-
from __future__ import print_function, division, absolute_import
import sys
from functools import partial
from itertools import chain
from collections import namedtuple

from numba.lexing import lex_source
from .annotate import format_annotations

WIDTH = 40
ANNOT_SEP = "-"

Emitter = namedtuple("Emitter", ["emit", "emitline"])
lex = partial(lex_source, output="console")

# ______________________________________________________________________

def render(annotation_blocks, emit=sys.stdout.write,
           intermediate_names=(), inline=True):
    """
    Render a Program as text.

    :param intermediate_names: [intermediate_name], e.g. ["llvm"]
    :param inline: whether to display intermediate code inline
    """
    indent = 8
    emitline = lambda indent, s: emit(u" " * indent + s + u"\n")
    emitter = Emitter(emit, emitline)

    for i, block in enumerate(annotation_blocks):
        python_source = block['python_source']
        intermediates = block['intermediates']

        if intermediates:
            irs = [i for i in intermediates if i.name in intermediate_names]
        else:
            irs = None

        # Render main source
        render_source(python_source, emitter, indent,
                      irs if inline and irs else [])

        if not inline and irs:
            # Render IRs seperately
            for irname, linenomap, ir_source in irs:
                emitter.emitline(0, irname.center(80, "="))
                render_source(ir_source, emitter, indent, [], linenomap)
            emitter.emitline(0, "=" * 80)

def render_source(source, emitter, indent, intermediates, linenomap=None):
    """
    Print a Source and its annotations. Print any given Intermediates inline.
    """
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
        line = lex(source.linemap[lineno])
        emitter.emitline(0, u"%4d    %s" % (lineno, line))

        annots = format_annotations(source.annotations.get(lineno, []))
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

def _gather_text_intermediates(intermediates, lineno):
    for irname, linenomap, ir_source in intermediates:
        ir_linenos = linenomap.get(lineno, [])
        if not ir_linenos:
            continue
        yield irname.center(WIDTH, "_")
        for ir_lineno in ir_linenos:
            yield lex(ir_source.linemap[ir_lineno], irname)