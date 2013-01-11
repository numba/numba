import ast
import copy
import opcode
import types
import logging
import __builtin__ as builtins

import numba
from numba import *
from numba import error, pipeline, nodes, ufunc_builder
from numba.minivect import minierror, minitypes, specializers, miniast
from numba import translate, utils, functions, nodes, transforms
from numba.symtab import Variable
from numba import visitors, nodes, error, ast_type_inference, ast_translate
from numba.utils import dump

from numbapro import vectorize, dispatch, array_slicing
from numbapro.vectorize import basic, minivectorize

import numpy

print_ufunc = False

class ArrayExpressionRewrite(visitors.NumbaTransformer,
                             ast_type_inference.NumpyMixin,
                             transforms.MathMixin):
    """
    Find element-wise expressions and run ElementalMapper to turn it into
    a minivect AST or a ufunc.
    """

    nesting_level = 0
    elementwise = False

    is_slice_assign = False

    def register_array_expression(self, node, lhs=None):
        """
        Start the mapping process for the outermost node in the array expression.
        """

    def visit_elementwise(self, elementwise, node):
        if elementwise and self.nesting_level == 0:
            return self.register_array_expression(node)

        self.nesting_level += 1
        self.generic_visit(node)
        self.nesting_level -= 1
        self.elementwise = elementwise
        return node


    def get_py_ufunc_ast(self, lhs, node):
        if lhs is not None:
            lhs.ctx = ast.Load()

        builder = ufunc_builder.UFuncConverter()
        tree = builder.visit(node)
        ufunc_ast = builder.build_ufunc_ast(tree)

        if print_ufunc:
            from meta import asttools
            module = ast.Module(body=[ufunc_ast])
            print asttools.python_source(module)

        # Vectorize Python function
        if lhs is None:
            restype = node.type
        else:
            restype = lhs.type.dtype

        argtypes = [op.type.dtype if op.type.is_array else op.type
                        for op in builder.operands]
        signature = restype(*argtypes)

        return ufunc_ast, signature, builder

    def get_py_ufunc(self, lhs, node):
        ufunc_ast, signature, ufunc_builder = self.get_py_ufunc_ast(lhs, node)
        py_ufunc = ufunc_builder.compile_to_pyfunc(ufunc_ast)
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

    def visit_MathNode(self, node):
        elementwise = node.arg.type.is_array
        return self.visit_elementwise(elementwise, node)

    def visit_BinOp(self, node):
        elementwise = node.type.is_array
        return self.visit_elementwise(elementwise, node)

    visit_UnaryOp = visit_BinOp


class ArrayExpressionRewriteUfunc(ArrayExpressionRewrite):
    """
    Compile array expressions to ufuncs. Then call the ufunc with the array
    arguments.

        vectorizer_cls: the ufunc vectorizer to use

    CANNOT be used in a nopython context
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

class ArrayExpressionRewriteNative(array_slicing.SliceRewriterMixin,
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

    CAN be used in a nopython context
    """

    def __init__(self, context, func, ast, llvm_module, **kwds):
        super(ArrayExpressionRewrite, self).__init__(context, func, ast, **kwds)
        self.array_expr_context = NumbaproStaticArgsContext()
        self.llvm_module = llvm_module
        self.array_expr_context.llvm_module = llvm_module

    def array_attr(self, node, attr):
        # Perform a low-level bitcast from object to an array type
        # array = nodes.CoercionNode(node, float_[:])
        array = node
        return nodes.ArrayAttributeNode(attr, array)

    def register_array_expression(self, node, lhs=None):
        lhs_type = lhs.type if lhs else node.type
        is_expr = lhs is None

        if lhs_type.ndim < node.type.ndim:
            # TODO: this is valid in NumPy if the leading dimensions of the
            # TODO: RHS have extent 1
            raise error.NumbaError(
                node, "Right hand side must have a "
                      "dimensionality <= %d" % lhs_type.ndim)

        # Create ufunc scalar kernel
        ufunc_ast, signature, ufunc_builder = self.get_py_ufunc_ast(lhs, node)
        signature.struct_by_reference = True

        # Compile ufunc scalar kernel with numba
        ast.fix_missing_locations(ufunc_ast)
        p, (_, _, _) = pipeline.run_pipeline(
                        self.context, None, ufunc_ast, signature, codegen=True)

        # Manual linking
        lfunc_name = p.translator.lfunc.name
        lfunc_type = p.translator.lfunc.type
        self.llvm_module.link_in(p.translator.lfunc.module)
        lfunc = self.llvm_module.get_function_named(lfunc_name)

        # print lfunc
        operands = ufunc_builder.operands
        self.func.live_objects.append(lfunc)

        operands = [nodes.CloneableNode(operand) for operand in operands]

        if lhs is not None:
            lhs = nodes.CloneableNode(lhs)
            broadcast_operands = [lhs] + operands
            lhs = lhs.clone
        else:
            broadcast_operands = operands[:]

        shape = array_slicing.BroadcastNode(lhs_type, broadcast_operands)
        operands = [op.clone for op in operands]

        if lhs is None and self.nopython:
            raise error.NumbaError(
                    node, "Cannot allocate new memory in nopython context")
        elif lhs is None:
            # TODO: determine best output order at runtime
            shape = shape.cloneable
            lhs = nodes.ArrayNewEmptyNode(lhs_type, shape.clone,
                                          lhs_type.is_f_contig).cloneable

        # Build minivect wrapper kernel
        context = self.array_expr_context
        # context.debug = True
        context.optimize_broadcasting = False
        b = context.astbuilder

        variables = [b.variable(name_node.type, "op%d" % i)
                         for i, name_node in enumerate([lhs] + operands)]
        miniargs = map(b.funcarg, variables)
        body = minivectorize.build_kernel_call(lfunc, signature, miniargs, b)

        minikernel = minivectorize.build_minifunction(body, miniargs, b)
        lminikernel, ctypes_kernel = context.run_simple(
                            minikernel, specializers.StridedSpecializer)

        # Build call to minivect kernel
        operands.insert(0, lhs)
        args = [shape]
        scalar_args = []
        for operand in operands:
            if operand.type.is_array:
                data_p = self.array_attr(operand, 'data')
                data_p = nodes.CoercionNode(data_p, operand.type.dtype.pointer())
                if not isinstance(operand, nodes.CloneNode):
                    operand = nodes.CloneNode(operand)
                strides_p = self.array_attr(operand, 'strides')
                args.extend((data_p, strides_p))
            else:
                scalar_args.append(operand)

        args.extend(scalar_args)
        result = nodes.NativeCallNode(minikernel.type, args, lminikernel)

        # Use native slicing in array expressions
        array_slicing.mark_nopython(ast.Suite(body=result.args))

        if not is_expr:
            # a[:] = b[:] * c[:]
            return result

        # b[:] * c[:], return new array as expression
        return nodes.ExpressionNode(stmts=[result], expr=lhs.clone)
