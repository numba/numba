# -*- coding: utf-8 -*-

"""
Numba annotators.
"""

from __future__ import print_function, division, absolute_import

import re

import numba
from numba import *
from numba import nodes

from numba.annotate import annotate
from numba.annotate.annotate import Annotation, A_c_api

logger = logging.getLogger(__name__)

# Taken from Cython/Compiler/Annotate.py
py_c_api = re.compile(u'(Py[A-Z][a-z]+_[A-Z][a-z][A-Za-z_]+)\(')
py_macro_api = re.compile(u'(Py[A-Z][a-z]+_[A-Z][A-Z_]+)\(')

# ______________________________________________________________________

class AnnotateTypes(object):

    def visit_Name(self, node):
        if isinstance(node, nodes.ExprNode):
            self.annotate(annotate.A_type, (node.id, str(node.type)))

# ______________________________________________________________________

def annotate_pyapi(llvm_intermediate, py_annotations):
    """
    Produce annotations for CPython C API calls from an LLVM SourceIntermediate
    """
    for py_lineno in llvm_intermediate.linenomap:
        count = 0
        for llvm_lineno in llvm_intermediate.linenomap[py_lineno]:
            line = llvm_intermediate.source.linemap[llvm_lineno]
            if re.search(py_c_api, line):
                count += 1

        if count:
            py_annotations[py_lineno].append(Annotation(A_c_api, count))
