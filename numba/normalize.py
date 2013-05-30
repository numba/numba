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

    function_level = 0

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

    def visit_Raise(self, node):
        raise error.NumbaError(node, "Raise statement not implemented yet")

    #------------------------------------------------------------------------
    # Normalization
    #------------------------------------------------------------------------

    def visit_FunctionDef(self, node):
        if self.function_level:
            return self.handle_inner_function(node)

        self.function_level += 1
        self.visitchildren(node)
        self.function_level -= 1
        return node

    def handle_inner_function(self, node):
        "Create assignment code for inner functions and mark the assignment"
        lhs = ast.Name(node.name, ast.Store())
        ast.copy_location(lhs, node)

        rhs = FuncDefExprNode(func_def=node)
        ast.copy_location(rhs, node)

        fields = rhs._fields
        rhs._fields = []
        assmnt = ast.Assign(targets=[lhs], value=rhs)
        result = self.visit(assmnt)
        rhs._fields = fields

        return result

    def visit_FunctionDef(self, node):
        #for arg in node.args:
        #    if arg.default:
        #        self.visitchildren(arg)
        if self.function_level:
            return self.handle_inner_function(node)

        self.visitchildren(node)
        return node

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
        list_create.type = typesystem.object_ # typesystem.list_()
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


    ######## Sync with cf2 branch in validators.py ###########

    def visit_With(self, node):
        node.context_expr = self.visit(node.context_expr)
        if node.optional_vars:
            raise error.NumbaError(
                node.context_expr,
                "Only 'with python' and 'with nopython' is "
                "supported at this moment")

        self.visitlist(node.body)
        return node


    def visit_For(self, node):
        if not isinstance(node.target, (ast.Name, ast.Attribute)):
            raise error.NumbaError(
                node.target, "Only a single target iteration variable is "
                             "supported at the moment")

        self.visitchildren(node)
        return node

#------------------------------------------------------------------------
# Nodes
#------------------------------------------------------------------------

class FuncDefExprNode(nodes.Node):
    """
    Wraps an inner function node until the closure code kicks in.
    """

    _fields = ['func_def']