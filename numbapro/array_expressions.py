import ast
import copy
import opcode
import types
import logging
import __builtin__ as builtins

import numba
from numba import *
from numba import error, pipeline, nodes
from numba.minivect import minierror, minitypes, specializers, miniast
from numba import translate, utils, functions, nodes
from numba.symtab import Variable
from numba import visitors, nodes, error, ast_type_inference, ast_translate
from numba import decorators

from numbapro import vectorize, dispatch, array_slicing
from numbapro.vectorize import basic, minivectorize

import numpy

class NumbaproPipeline(pipeline.Pipeline):
    def __init__(self, context, func, ast, func_signature, **kwargs):
        super(NumbaproPipeline, self).__init__(context, func, ast,
                                               func_signature, **kwargs)
        self.insert_specializer('rewrite_array_expressions',
                                after='specialize')

    def rewrite_array_expressions(self, ast):
        transformer = ArrayExpressionRewriteUfunc(self.context, self.func, ast)
        transformer = ArrayExpressionRewriteGPU(self.context, self.func, ast,
                                                self.llvm_module)
        return transformer.visit(ast)

    def codegen(self, ast):
        self.translator = self.make_specializer(ArrayExpressionGPUCodegen,
                                                ast, func_name=self.func_name,
                                                **self.kwargs)
        self.translator.translate()
        return ast

decorators.context.numba_pipeline = NumbaproPipeline


class ArrayExpressionRewrite(visitors.NumbaTransformer,
                             ast_type_inference.NumpyMixin):
    """
    Find element-wise expressions and run ElementalMapper to turn it into
    a minivect AST or a ufunc.
    """

    is_slice_assign = False
    nesting_level = 0
    elementwise = False

    def register_array_expression(self, node, lhs=None):
        """
        Start the mapping process for the outmost node in the array expression.
        """

    def get_py_ufunc(self, lhs, node):
        if lhs is None:
            result_type = node.type
        else:
            result_type = lhs.type
            lhs.ctx = ast.Load()

        # Build ufunc AST module
        ufunc_builder = UFuncBuilder(self.context, self.func, node)
        tree = ufunc_builder.visit(node)
        ufunc_ast = ufunc_builder.build_ufunc_ast(tree)
        module = ast.Module(body=[ufunc_ast])
        functions.fix_ast_lineno(module)
        # logging.debug(ast.dump(ufunc_ast))

        # Create Python ufunc function
        d = {}
        exec compile(module, '<ast>', 'exec') in d, d
        py_ufunc = d['ufunc']

        # Vectorize Python function
        if lhs is None:
            restype = node.type
        else:
            restype = lhs.type.dtype

        argtypes = [op.type.dtype if op.type.is_array else op.type
                        for op in ufunc_builder.operands]
        signature = restype(*argtypes)
        return py_ufunc, signature, ufunc_builder

    def visit_Assign(self, node):
        self.is_slice_assign = False
        self.visitlist(node.targets)
        is_slice_assign = self.is_slice_assign

        self.nesting_level = self.is_slice_assign
        node.value = self.visit(node.value)
        self.nesting_level = 0

        elementwise = self.elementwise
        if (len(node.targets) == 1 and node.targets[0].type.is_array and
                is_slice_assign and elementwise):
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
        elementwise = node.type.is_array
        if elementwise and self.nesting_level == 0:
            return self.register_array_expression(node)

        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1

        self.elementwise = elementwise
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


class ArrayExpressionRewriteUfunc(ArrayExpressionRewrite):
    """
    Compile array expressions to ufuncs. Then call the ufunc with the array
    arguments.

    vectorizer_cls: the ufunc vectorizer to use
    """

    def __init__(self, context, func, ast, vectorizer_cls=None):
        super(ArrayExpressionRewriteUfunc, self).__init__(context, func, ast)
        self.vectorizer_cls = vectorizer_cls or basic.BasicASTVectorize

    def register_array_expression(self, node, lhs=None):
        py_ufunc, signature, ufunc_builder = self.get_py_ufunc(lhs, node)

        # Vectorize Python function
        vectorizer = self.vectorizer_cls(py_ufunc)
        vectorizer.add(restype=signature.return_type, argtypes=signature.args)
        ufunc = vectorizer.build_ufunc()

        # Call ufunc
        args = ufunc_builder.operands
        if lhs is None:
            keywords = None
        else:
            keywords = [ast.keyword('out', lhs)]

        func = nodes.ObjectInjectNode(ufunc)
        call_ufunc = nodes.ObjectCallNode(signature=None, func=func, args=args,
                                          keywords=keywords, py_func=ufunc)
        return nodes.ObjectTempNode(call_ufunc)


class NumbaproStaticArgsContext(utils.NumbaContext):
    "Use a static argument list: shape, data1, strides1, data2, strides2, ..."
    astbuilder_cls = miniast.ASTBuilder

class ArrayExpressionRewriteGPU(array_slicing.SliceRewriterMixin,
                                ArrayExpressionRewrite):
    """
    Compile array expressions to a minivect kernel that calls a Numba
    scalar kernel with scalar inputs:

        a[:, :] = b[:, :] * c[:, :]

    becomes

        tmp_a = slice(a)
        tmp_b = slice(b)
        tmp_c = slice(c)
        shape = broadcast(tmp_a, tmp_b, tmp_c)
        call minikernel(shape, tmp_a.data, tmp_a.strides,
                               tmp_b.data, tmp_b.strides,
                               tmp_c.data, tmp_c.strides)

    with

        def numba_kernel(b, c):
            return b * c

        def minikernel(...):
            for (...)
                for(...)
                    a[i, j] = numba_kernel(b[i, j], c[i, j])
    """

    def __init__(self, context, func, ast, llvm_module):
        super(ArrayExpressionRewriteGPU, self).__init__(context, func, ast)
        self.array_expr_context = NumbaproStaticArgsContext()
        self.llvm_module = llvm_module
        self.array_expr_context.llvm_module = llvm_module

    def array_attr(self, node, attr):
        return nodes.ArrayAttributeNode(attr, node)

    def register_array_expression(self, node, lhs=None):
        # Create ufunc scalar kernel
        py_ufunc, signature, ufunc_builder = self.get_py_ufunc(lhs, node)
        # Compile ufunc scalar kernel with numba
        sig, translator, wrapper = pipeline.compile_from_sig(
                    self.context, py_ufunc, signature, compile_only=True)

        lfunc = translator.lfunc
        operands = ufunc_builder.operands
        self.func.live_objects.append(lfunc)

        if lhs is None:
            raise error.NumbaError(node, "Cannot allocate new memory on GPU")

        # Build minivect wrapper kernel
        context = self.array_expr_context
        # context.debug = True
        b = context.astbuilder

        variables = [b.variable(name_node.type, "op%d" % i)
                         for i, name_node in enumerate([lhs] + operands)]
        miniargs = map(b.funcarg, variables)
        body = minivectorize.build_kernel_call(lfunc, signature, miniargs, b)

        minikernel = minivectorize.build_minifunction(body, miniargs, b)
        lminikernel, ctypes_kernel = context.run_simple(
                            minikernel, specializers.StridedSpecializer)

        # Build call to minivect kernel
        operands = [nodes.CloneableNode(operand) for operand in operands]

        lhs = nodes.CloneableNode(lhs)
        # The broadcast shape. There is no error checking or actual
        # broadcasting!
        # Broadcasting dimensions are assumed to have stride 0, this may
        # not be true at all!
        shape = self.array_attr(lhs, 'shape')

        lhs = nodes.CloneNode(lhs)
        operands.insert(0, lhs)

        args = [shape]
        for node in operands:
            data_p = self.array_attr(node, 'data')
            args.append(nodes.CoercionNode(data_p, operand.type.dtype.pointer()))
            if not isinstance(node, nodes.CloneNode):
                node = nodes.CloneNode(node)
            args.append(self.array_attr(node, 'strides'))

        return nodes.NativeCallNode(minikernel.type, args, lminikernel)

class ArrayExpressionGPUCodegen(ast_translate.LLVMCodeGenerator,
                                array_slicing.NativeSliceCodegenMixin):
    pass
