# -*- coding: UTF-8 -*-

"""
numba --annotate
"""

from __future__ import print_function, division, absolute_import

import operator
from itertools import groupby
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
   
    func_code = getattr(func, 'func_code', None)
    if func_code is None:
        func_code = getattr(func, '__code__')
    lineno = func_code.co_firstlineno
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

# Annotation formatting

def format_annotations(annotations):
    adict = groupdict(annotations, 'type')
    for category, annotations in adict.items():
        vals = u" ".join([str(a.value) for a in annotations])
        yield u"%s: %s" % (category, vals)

groupdict = lambda xs, attr: dict(
    (k, list(v)) for k, v in groupby(xs, operator.attrgetter(attr)))

# ______________________________________________________________________