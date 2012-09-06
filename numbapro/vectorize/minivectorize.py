import ast
import ctypes

from numba import visitors as numba_visitors
from numba.minivect import (miniast,
                            minitypes,
                            specializers as minispecializers,
                            ctypes_conversion)
from numba import decorators, utils, functions

from numbapro import _internal, _minidispatch
from numbapro.vectorize import _common, basic

import numpy as np

debug_c = False
debug_llvm = False

minicontext = decorators.context

class UntranslatableError(Exception):
    pass

opmap = {
    # Unary
    ast.Invert: '~',
    ast.Not: None, # not supported
    ast.UAdd: '+',
    ast.USub: '-',

    # Binary
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Mod: '%',
    ast.Pow: '**',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.BitAnd: '&',
    ast.FloorDiv: '//',

    # Comparison
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Is: None,
    ast.IsNot: None,
    ast.In: None,
    ast.NotIn: None,
}

def getop(ast_op):
    ast_op = type(ast_op)
    if opmap[ast_op] is None:
        raise UnicodeTranslateError("Invalid element-wise operator")
    return opmap[ast_op]

class PythonUfunc2Minivect(numba_visitors.NumbaVisitor):
    def __init__(self, context, func, func_ast, ret_type, arg_types):
        super(PythonUfunc2Minivect, self).__init__(context, func, ast)

        if self.global_names:
            raise UntranslatableError("Function uses globals")

        assert isinstance(func_ast, ast.FunctionDef)

        self.ret_type = ret_type
        self.arg_types = arg_types

        self.b = context.astbuilder
        self.lhs = self.b.variable(ret_type, 'lhs')

        self.variables = dict(
            (arg_name, self.b.variable(arg_type, arg_name))
                for arg_type, arg_name in zip(arg_types, self.argnames))

        variables = [self.lhs] + [self.variables[arg_name]
                                      for arg_name in self.argnames]
        self.miniargs = [self.b.funcarg(variable) for variable in variables]
        self.seen_return = False

    def visit_FunctionDef(self, node):
        return self.b.stats(*self.visitlist(node.body))

    def visit_Name(self, node):
        return self.variables[node.id]

    def visit_Return(self, node):
        if self.seen_return:
            raise UntranslatableError("multiple return statements")

        self.seen_return = True
        return self.b.assign(self.lhs, self.visit(node.value))

    def visit_BinOp(self, node):
        lhs, rhs = self.visit(node.left), self.visit(node.right)
        type = self.context.promote_types(lhs.type, rhs.type)
        return self.b.binop(type, getop(node.op), lhs, rhs)

    def visit_Unop(self, node):
        return self.b.unop(node.type, getop(node.op), self.visit(node.operand))

    def generic_visit(self, node):
        raise UntranslatableError("Unsupported node of type %s" %
                                                    type(node).__name__)

class NumbaContigSpecializer(minispecializers.ContigSpecializer):
    def visit_FunctionNode(self, node):
        node = super(minispecializers.ContigSpecializer,
                     self).visit_FunctionNode(node)
        self.astbuilder.create_function_type(node, strides_args=True)
        return node

    def visit_StridePointer(self, node):
        self.visitchildren(node)
        return node

class MiniVectorize(object):

    specializers = [
        NumbaContigSpecializer,
        minispecializers.StridedCInnerContigSpecializer,
        minispecializers.CTiledStridedSpecializer,
        minispecializers.StridedSpecializer,
    ]

    num_exprs = 0

    def __init__(self, func, fallback=basic.BasicVectorize):
        self.pyfunc = func
        self.ast = functions._get_ast(func)
        self.signatures = []
        self.fallback = fallback

    def add(self, ret_type, arg_types, **kwargs):
        self.signatures.append((ret_type, arg_types, kwargs))

    def build_ufunc(self):
        minivect_asts = []
        for ret_type, arg_types, kwargs in self.signatures:
            for dimensionality in (1, 2):
                array_types = [minitypes.ArrayType(arg_type, dimensionality)
                                   for arg_type in arg_types]
                rtype = minitypes.ArrayType(ret_type, dimensionality)
                mapper = PythonUfunc2Minivect(decorators.context, self.pyfunc,
                                              self.ast, rtype, array_types)
                try:
                    minivect_ast = mapper.visit(self.ast)
                except UntranslatableError, e:
                    # TODO: translate with Numba, call from minivect (non-inlined)
                    print e
                    return self.fallback_vectorize(None)
                else:
                    minivect_asts.append((mapper, dimensionality, minivect_ast))

        # return self.minivect(minivect_asts)
        return self.fallback_vectorize(self.minivect(minivect_asts))

    def build_minifunction(self, ast, miniargs):
        b = minicontext.astbuilder

        type = minitypes.npy_intp.pointer()
        shape_variable = b.variable(type, 'shape')

        minifunc = b.function('expr%d' % MiniVectorize.num_exprs,
                              ast, miniargs, shapevar=shape_variable)
        MiniVectorize.num_exprs += 1

        # minifunc.print_tree(minicontext)

        return minifunc

    def minivect(self, asts):
        """
        Given a bunch of specialized miniasts, return a ufunc object that
        invokes the right specialization when called.
        """
        ufuncs = {}
        for mapper, dimensionality, ast in asts:
            minifunc = self.build_minifunction(ast, mapper.miniargs)
            if debug_c:
                print minicontext.debug_c(minifunc,
                                          minispecializers.StridedCInnerContigSpecializer)
            result = list(minicontext.run(minifunc, self.specializers))

            # Map minitypes to NumPy dtypes, so we can find the specialization
            # given input NumPy arrays
            result_dtype = minitypes.map_minitype_to_dtype(mapper.ret_type.dtype)
            dtype_args = [minitypes.map_minitype_to_dtype(arg_type)
                              for arg_type in mapper.arg_types]

            llvm_funcs, ctypes_funcs = zip(*[
                    (lfunc, ctypes_func)
                        for _, _, _, (lfunc, ctypes_func) in result])
            function_pointers = [
                ctypes_conversion.get_pointer(minicontext, llvm_func)
                    for llvm_func in llvm_funcs]

            if debug_llvm:
                lfunc = result[-1][3][0]
                print lfunc

            # Get the argument ctypes types so we can cast the NumPy data pointer
            ctypes_ret_type = ctypes_conversion.convert_to_ctypes(
                                    mapper.ret_type.dtype.pointer())
            ctypes_arg_types = []
            for arg_type in mapper.arg_types:
                dtype_pointer = arg_type.dtype.pointer()
                ctypes_arg_types.append(
                    ctypes_conversion.convert_to_ctypes(dtype_pointer))

            dtype_args.append(dimensionality)
            ufuncs[tuple(dtype_args)] = (function_pointers,
                                         ctypes_funcs, ctypes_ret_type,
                                         ctypes_arg_types, result_dtype)

        return _minidispatch.UFuncDispatcher(ufuncs, len(mapper.arg_types))

    def fallback_vectorize(self, minivect_dispatcher):
        vectorizer = self.fallback(self.pyfunc)
        for ret_type, arg_types, kwargs in self.signatures:
            vectorizer.add(ret_type=ret_type, arg_types=arg_types, **kwargs)

        return vectorizer.build_ufunc(minivect_dispatcher)


if __name__ == '__main__':
    import time

    def vector_add(a, b):
        return a + b

    vectorizer = MiniVectorize(vector_add)
    f32 = minitypes.float32
    vectorizer.add(ret_type=f32, arg_types=[f32, f32])
    ufunc = vectorizer.build_ufunc()

    dtype = np.float32
    N = 4000

    a = np.arange(N * N, dtype=dtype).reshape(N, N)[1:-1, 1:-1]
    b = a.copy() #.T
    out = np.empty_like(a)

    ufunc(a, b, out=out)
    assert np.all(out == a + b)

    t = time.time()
    for i in range(100):
        ufunc(a, b, out=out)
    print time.time() - t
