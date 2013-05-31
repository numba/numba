# -*- coding: utf-8 -*-

"""
Numba annotators.
"""

from __future__ import print_function, division, absolute_import

import sys
import ast

import numba
from numba import *
from numba import error
from numba import nodes

from numba.annotate import annotate
from numba.annotate.annotate import (Source, Annotation, Intermediate, Program,
                                     A_type, render_text, Renderer)

import llvm.core
import numpy as np

logger = logging.getLogger(__name__)

class AnnotateTypes(object):

    def visit_Name(self, node):
        if isinstance(node, nodes.ExprNode):
            self.annotate(annotate.A_type, (node.id, str(node.type)))