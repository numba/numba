# -*- coding: utf-8 -*-

"""
Initial AST validation and normalization.
"""

from __future__ import print_function, division, absolute_import

import ast
import copy

from numba import error
from numba import nodes
from numba import visitors
from numba import typesystem


class NormalizeAST(visitors.NumbaTransformer):
    "Normalize AST"

    def visit_GeneratorExp(self, node):
        raise error.NumbaError(
                node, "Generator comprehensions are not yet supported")

    def visit_SetComp(self, node):
        raise error.NumbaError(
                node, "Set comprehensions are not yet supported")

    def visit_DictComp(self, node):
        raise error.NumbaError(
                node, "Dict comprehensions are not yet supported")

    def visit_Raise(self, node):
        raise error.NumbaError(node, "Raise statement not implemented yet")

    def visit_ListComp(self, node):
        """
        Rewrite list comprehensions to the equivalent for loops.

        AST syntax:

            ListComp(expr elt, comprehension* generators)
            comprehension = (expr target, expr iter, expr* ifs)

            'ifs' represent a chain of ANDs
        """
        assert len(node.generators) > 0

        # Create innermost body, i.e. list.append(expr)
        # TODO: size hint for PyList_New
        list_create = ast.List(elts=[], ctx=ast.Load())
        list_create.type = typesystem.object_ # typesystem.ListType()
        list_create = nodes.CloneableNode(list_create)
        list_value = nodes.CloneNode(list_create)
        list_append = ast.Attribute(list_value, "append", ast.Load())
        append_call = ast.Call(func=list_append, args=[node.elt],
                               keywords=[], starargs=None, kwargs=None)

        # Build up the loops from inwards to outwards
        body = append_call
        for comprehension in reversed(node.generators):
            # Hanlde the 'if' clause
            ifs = comprehension.ifs
            if len(ifs) > 1:
                make_boolop = lambda op1_op2: ast.BoolOp(op=ast.And(),
                                                         values=op1_op2)
                if_test = reduce(make_boolop, ifs)
            elif len(ifs) == 1:
                if_test, = ifs
            else:
                if_test = None

            if if_test is not None:
                body = ast.If(test=if_test, body=[body], orelse=[])

            # Wrap list.append() call or inner loops
            body = ast.For(target=comprehension.target,
                           iter=comprehension.iter, body=[body], orelse=[])

        expr = nodes.ExpressionNode(stmts=[list_create, body], expr=list_value)
        return self.visit(expr)

    def visit_AugAssign(self, node):
        """
        Inplace assignment.

        Resolve a += b to a = a + b. Set 'inplace_op' attribute of the
        Assign node so later stages may recognize inplace assignment.

        Do this now, so that we can correctly mark the RHS reference.
        """
        target = node.target

        rhs_target = copy.deepcopy(target)
        rhs_target.ctx = ast.Load()
        ast.fix_missing_locations(rhs_target)

        bin_op = ast.BinOp(rhs_target, node.op, node.value)
        assignment = ast.Assign([target], bin_op)
        assignment.inplace_op = node.op
        return self.visit(assignment)
