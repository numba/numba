# -*- coding: utf-8 -*-

"""
Initial AST validation and normalization.
"""

from __future__ import print_function, division, absolute_import

import ast
from numba import error, nodes

class ValidateAST(ast.NodeVisitor):
    "Validate AST"

    #------------------------------------------------------------------------
    # Validation
    #------------------------------------------------------------------------

    def visit_GeneratorExp(self, node):
        raise error.NumbaError(
                node, "Generator comprehensions are not yet supported")

    def visit_SetComp(self, node):
        raise error.NumbaError(
                node, "Set comprehensions are not yet supported")

    def visit_DictComp(self, node):
        raise error.NumbaError(
                node, "Dict comprehensions are not yet supported")

    # def visit_Raise(self, node):
    #     if node.tback:
    #         raise error.NumbaError(
    #             node, "Traceback argument to raise not supported")

    def visit_For(self, node):
        if not isinstance(node.target, (ast.Name, ast.Attribute,
                                        nodes.TempStoreNode)):
            raise error.NumbaError(
                node.target, "Only a single target iteration variable is "
                             "supported at the moment")

        self.generic_visit(node)

    def visit_With(self, node):
        self.visit(node.context_expr)
        if node.optional_vars:
            raise error.NumbaError(
                node.context_expr,
                "Only 'with python' and 'with nopython' is "
                "supported at this moment")

        self.generic_visit(node)