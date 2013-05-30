# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import ast
import types

from numba import visitors, nodes, error, functions, traits

class UFuncBuilder(object):
    """
    Create a Python ufunc AST function. Demote the types of arrays to scalars
    in the ufunc and generate a return.
    """

    ufunc_counter = 0

    def __init__(self):
        self.operands = []

    def register_operand(self, node):
        """
        Register a sub-expression as something that will be evaluated
        outside the kernel, and the result of which will be passed into the
        kernel. This can be a variable name:

            a + b

        ->

            f(arg1, arg2):
                return arg1 + arg2

        For which 'a' and 'b' are the operands.
        """
        result = ast.Name(id='op%d' % len(self.operands), ctx=ast.Load())
        self.operands.append(node)
        return result

    def build_ufunc_ast(self, tree):
        args = [ast.Name(id='op%d' % i, ctx=ast.Param())
                    for i, op in enumerate(self.operands)]
        arguments = ast.arguments(args=args,
                                  vararg=None,
                                  kwarg=None,
                                  defaults=[],
        )
        body = ast.Return(value=tree)
        func = ast.FunctionDef(name='ufunc%d' % self.ufunc_counter,
                               args=arguments, body=[body], decorator_list=[])
        UFuncBuilder.ufunc_counter += 1
        # print ast.dump(func)
        return func

    def compile_to_pyfunc(self, ufunc_ast, globals=()):
        "Compile the ufunc ast to a function"
        # Build ufunc AST module
        module = ast.Module(body=[ufunc_ast])
        functions.fix_ast_lineno(module)

        # Create Python ufunc function
        d = dict(globals)
        exec(compile(module, '<ast>', 'exec'), d, d)
        d.pop('__builtins__')
        py_ufunc = d[ufunc_ast.name]

        assert isinstance(py_ufunc, types.FunctionType), py_ufunc

        return py_ufunc

    def save(self):
        """
        Save the state of the builder to allow processing other parts of
        the tree.
        """
        state = self.operands
        self.operands = []
        return state

    def restore(self, state):
        "Restore saved state"
        self.operands = state

@traits.traits
class UFuncConverter(ast.NodeTransformer):
    """
    Convert a Python array expression AST to a scalar ufunc kernel by demoting
    array types to scalar types.
    """

    build_ufunc_ast = traits.Delegate('ufunc_builder')
    operands = traits.Delegate('ufunc_builder')

    def __init__(self, env):
        self.ufunc_builder = UFuncBuilder()
        self.env = env

    def demote_type(self, node):
        node.type = self.demote(node.type)
        if hasattr(node, 'variable'):
            node.variable.type = node.type

    def demote(self, type):
        if type.is_array:
            return type.dtype
        return type

    def visit_BinOp(self, node):
        if node.type.is_array:
            self.demote_type(node)
            node.left = self.visit(node.left)
            node.right = self.visit(node.right)
        else:
            node = self.generic_visit(node)
        return node

    def visit_UnaryOp(self, node):
        self.demote_type(node)
        node.operand = self.visit(node.operand)
        return node

    def visit_Call(self, node):
        if nodes.query(self.env, node, "is_math") and node.type.is_array:
            self.demote_type(node)
            node.args = list(map(self.visit_scalar_or_array, node.args))
        else:
            node = self.generic_visit(node)
        return node

    def visit_CoercionNode(self, node):
        return self.visit(node.node)

    def _generic_visit(self, node):
        super(UFuncBuilder, self).generic_visit(node)

    def visit_scalar_or_array(self, node):
        if node.type.is_array:
            node =  self.visit(node)
            self.demote_type(node)
            return node
        else:
            return self.generic_visit(node)

    def generic_visit(self, node):
        """
        Register Name etc as operands to the ufunc
        """
        result = self.ufunc_builder.register_operand(node)
        result.type = node.type
        self.demote_type(result)
        return result
