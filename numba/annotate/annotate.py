# -*- coding: UTF-8 -*-

"""
numba --annotate
"""

from __future__ import print_function, division, absolute_import

import sys
import operator
from itertools import groupby
from collections import namedtuple

# ______________________________________________________________________

Program = namedtuple("Program", ["python_source", "intermediates"])
Intermediate = namedtuple("Intermediate", ["name", "pyline2line", "source"])
Source = namedtuple("Source", ["linemap", "annotations"])
Annotation = namedtuple("Annotation", ["type", "value"])

# ______________________________________________________________________

class Renderer(object):
    """
    Render an intermediate source.

    Capabilities: Set of capabilities as strings. Capabilities include:

        "dot":
            return a graphviz dot representation as string
        "source":
            return a two-tuple of ({ python lineno : source lineno }, Source)
    """

    capabilities = frozenset()

    def render(self, capability):
        raise NotImplementedError

# ______________________________________________________________________

# Annotation types

A_type      = "Types"
A_c_api     = "Python C API"
A_numpy     = "NumPy"
A_errcheck  = "Error check"
A_objcoerce = "Coercion"
A_pycall    = "Python call"
A_pyattr    = "Python attribute"

# ______________________________________________________________________

groupdict = lambda xs, attr: dict(
    (k, list(v)) for k, v in groupby(xs, operator.attrgetter(attr)))

def render_text(program, emit=sys.stdout.write, capabilities=frozenset(),
                inline=True):
    iemit = lambda indent, s: emit(u" " * indent + s)
    emitline = lambda indent, s: iemit(indent, s + u"\n")
    indent = 8

    pysrc = program.python_source
    maxlength = max(map(len, pysrc.linemap.values()))

    for lineno, sourceline in pysrc.linemap.items():
        # Print python source line
        emitline(0, u"%4d    %s" % (lineno, sourceline))

        if lineno in pysrc.annotations:
            # Gather annotation lines: 'Category: annotation values"
            annotations = pysrc.annotations[lineno]
            adict = groupdict(annotations, 'type')

            lines = []
            for category, annotations in adict.items():
                vals = u" ".join([str(a.value) for a in annotations])
                lines.append(u"| %s: %s" % (category, vals))

            if lines:
                # Print out annotations
                linestart = indent + len(sourceline) - len(sourceline.lstrip())
                # emitline(linestart, "_" * (maxlength - (linestart - indent)))
                emitline(linestart + 2, u"========||========")
                for line in lines:
                    emitline(linestart + 2, line)
                emitline(linestart + 2, u"========||========")

# ______________________________________________________________________