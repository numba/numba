import ast
import copy
import opcode
import types
import logging
import __builtin__ as builtins

import numba
from numba import *
from numba import error, pipeline
from numba.minivect import minierror, minitypes
from numba import translate, utils, functions, nodes
from numba.symtab import Variable
from numba import visitors, nodes, error, ast_type_inference
from numba import decorators

from numbapro import vectorize
from numbapro.vectorize import basic

import numpy

class NumbaproPipeline(pipeline.Pipeline):
    def __init__(self, context, func, ast, func_signature, **kwargs):
        super(NumbaproPipeline, self).__init__(context, func, ast,
                                               func_signature, **kwargs)
        self.insert_specializer('rewrite_array_expressions',
                                after='specialize')

    def rewrite_array_expressions(self, ast):
        return UFuncRewriter(self.context, self.func, ast).visit(ast)

decorators.context.numba_pipeline = NumbaproPipeline


class ArrayExpressionRewrite(visitors.NumbaTransformer,
                             ast_type_inference.NumpyMixin):
    """
    Find element-wise expressions and run ElementalMapper to turn it into
    a minivect AST or a ufunc.
    """

    is_slice_assign = False
    nesting_level = 0

    def register_array_expression(self, node, lhs=None):
        """
        Start the mapping process for the outmost node in the array expression.
        """

    def visit_Assign(self, node):
        self.is_slice_assign = False
        self.visitlist(node.targets)

        self.nesting_level = self.is_slice_assign
        node.value = self.visit(node.value)
        self.nesting_level = 0

        elementwise = getattr(node.value, 'elementwise', False)
        if (len(node.targets) == 1 and node.targets[0].type.is_array and
                self.is_slice_assign and elementwise):
            return self.register_array_expression(node.value,
                                                  lhs=node.targets[0])

        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        self.is_slice_assign = (isinstance(node.ctx, ast.Store) and
                                node.type.is_array)
        if self._is_ellipsis(node.slice):
            return node.value
        return node

    def visit_BinOp(self, node):
        node.elementwise = node.type.is_array
        if node.elementwise and self.nesting_level == 0:
            return self.register_array_expression(node)

        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        return node

    visit_UnaryOp = visit_BinOp

ufunc_count = 0
class UFuncBuilder(visitors.NumbaTransformer):
    """
    Create a Python ufunc AST function. Demote the types of arrays to scalars
    in the ufunc and generate a return.
    """

    def __init__(self, context, func, ast):
        super(UFuncBuilder, self).__init__(context, func, ast)
        self.operands = []

    def demote_type(self, node):
        if node.type.is_array:
            node.type = node.type.dtype

    def visit_BinOp(self, node):
        self.demote_type(node)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node):
        self.demote_type(node)
        node.op = self.visit(node.op)
        return node

    def visit_CoercionNode(self, node):
        return self.visit(node.node)

    def _generic_visit(self, node):
        super(UFuncBuilder, self).generic_visit(node)

    def generic_visit(self, node):
        """
        Register Name etc as operands to the ufunc
        """
        self.operands.append(node)
        result = ast.Name(id='op%d' % (len(self.operands) - 1), ctx=ast.Load())
        result.type = node.type
        self.demote_type(result)
        return result

    def build_ufunc_ast(self, tree):
        args = [ast.Name(id='op%d' % i, ctx=ast.Param())
                    for i, op in enumerate(self.operands)]
        arguments = ast.arguments(args, # args
                                  None, # vararg
                                  None, # kwarg
                                  [],   # defaults
        )
        body = ast.Return(value=tree)
        func = ast.FunctionDef(name='ufunc', args=arguments,
                               body=[body], decorator_list=[])
        # print ast.dump(func)
        return func


class UFuncRewriter(ArrayExpressionRewrite):
    """
    Compile array expressions to ufuncs.

    vectorizer_cls: the ufunc vectorizer to use
    """

    def __init__(self, context, func, ast, vectorizer_cls=None):
        super(UFuncRewriter, self).__init__(context, func, ast)
        self.vectorizer_cls = vectorizer_cls or basic.BasicASTVectorize

    def register_array_expression(self, node, lhs=None):
        if lhs is None:
            result_type = node.type
        else:
            result_type = lhs.type

        # Build ufunc AST module
        ufunc_builder = UFuncBuilder(self.context, self.func, node)
        tree = ufunc_builder.visit(node)
        ufunc_ast = ufunc_builder.build_ufunc_ast(tree)
        module = ast.Module(body=[ufunc_ast])
        functions.fix_ast_lineno(module)

        logging.debug(ast.dump(ufunc_ast))

        # Create Python ufunc function
        d = {}
        exec compile(module, '<ast>', 'exec') in d, d
        py_ufunc = d['ufunc']

        # Vectorize Python function
        if lhs is None:
            restype = node.type
        else:
            restype = lhs.type.dtype

        vectorizer = self.vectorizer_cls(py_ufunc)
        argtypes = [op.type.dtype if op.type.is_array else op.type
                         for op in ufunc_builder.operands]
        vectorizer.add(restype=restype, argtypes=argtypes)
        ufunc = vectorizer.build_ufunc()

        # Call ufunc
        signature = minitypes.FunctionType(
                return_type=result_type,
                args=[op.type for op in ufunc_builder.operands])

        args = ufunc_builder.operands
        if lhs is None:
            keywords = None
        else:
            keywords = [ast.keyword('out', lhs)]

        func = nodes.ObjectInjectNode(ufunc)
        call_ufunc = nodes.ObjectCallNode(signature=signature,
                                          func=func, args=args,
                                          keywords=keywords, py_func=ufunc)
        return nodes.ObjectTempNode(call_ufunc)
