"""
Work in progress, do not use.
"""

__all__ = ['MiniVectorize']

import ast
import ctypes
import logging

from numba import visitors as numba_visitors
from numba.minivect import (miniast,
                            minitypes,
                            minierror,
                            specializers as minispecializers,
                            ctypes_conversion)
from numba import decorators, utils, functions

from _numba_vectorize import _internal
from numbapro import dispatch
from numbapro.vectorize import _common, basic, parallel

import numpy as np

debug_c = False
debug_llvm = False

minicontext = decorators.context
b = minicontext.astbuilder

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
    "Map AST binary operators to string binary operators (which minivect uses)"
    ast_op = type(ast_op)
    if opmap[ast_op] is None:
        raise UnicodeTranslateError("Invalid element-wise operator")
    return opmap[ast_op]

def fallback_vectorize(fallback_cls, pyfunc, signatures, dispatcher, **kw):
    """
    Build an actual ufunc, but dispatch to the given dispatcher for
    element-wise operations.

    Extra keyword arguments include
        module: the llvm module to compile in
        engine: the llvm execution engine to use
    """
    vectorizer = fallback_cls(pyfunc)
    for restype, argtypes, kwargs in signatures:
        kwargs = dict(kwargs, **kw)
        vectorizer.add(restype=restype, argtypes=argtypes, **kwargs)

    ufunc = vectorizer.build_ufunc(dispatcher)
    return ufunc, vectorizer._get_lfunc_list()

def build_kernel_call(lfunc, signature, miniargs, b=b):
    """
    Call the kernel `lfunc` in a bunch of loops with scalar arguments.
    """
    # Build the kernel function signature
    funcname = b.funcname(signature, lfunc.name, is_external=False)

    # Generate 'lhs[i, j] = kernel(A[i, j], B[i, j])'
    lhs = miniargs[0].variable
    kernel_args = [arg.variable for arg in miniargs[1:]]
    funccall = b.funccall(funcname, kernel_args, inline=True)
    assmt = b.assign(lhs, funccall)
    if lhs.type.is_object:
        assmt = b.stats(b.decref(lhs), assmt)

    return assmt

def build_minifunction(ast, miniargs, b=b):
    """
    Build a minivect function from the given ast and arguments.
    """
    type = minitypes.npy_intp.pointer()
    shape_variable = b.variable(type, 'shape')

    minifunc = b.function('expr%d' % MiniVectorize.num_exprs,
                          ast, miniargs, shapevar=shape_variable)
    MiniVectorize.num_exprs += 1

    # minifunc.print_tree(minicontext)

    return minifunc

def specialize(minifunc):
    "Specialize a minivect function"
    if minifunc.ndim < 2:
        # Remove tiled specializer
        specializers = [
            NumbaContigSpecializer,
            minispecializers.StridedSpecializer,
        ]
    else:
        specializers = [
            minispecializers.StridedCInnerContigSpecializer,
            minispecializers.CTiledStridedSpecializer,
            minispecializers.StridedSpecializer,
        ]

    result = list(minicontext.run(minifunc, specializers))

    contig = inner_contig = tiled = strided = None
    if minifunc.ndim < 2:
        contig, strided = result
    else:
        inner_contig, tiled, strided = result

    return [contig, inner_contig, tiled, strided]

class PythonUfunc2Minivect(numba_visitors.NumbaVisitor):
    """
    Map and inline a ufunc written in Python to a minivect AST. The result
    should be wrapped in a minivect function.

    Raises UntranslatableError in case it cannot convert the expression. In
    this case we could translate the ufunc using Numba, and create a minivect
    kernel that calls this function (in the same LLVM module), and uses LLVM
    to inline the Numba-compiled function.
    """

    def __init__(self, context, func, func_ast, restype, argtypes):
        super(PythonUfunc2Minivect, self).__init__(context, func, ast)
        assert isinstance(func_ast, ast.FunctionDef)

        self.restype = restype
        self.argtypes = argtypes

        self.b = context.astbuilder
        self.lhs = self.b.variable(restype, 'lhs')

        self.variables = dict(
            (arg_name, self.b.variable(arg_type, arg_name))
                for arg_type, arg_name in zip(argtypes, self.argnames))

        variables = [self.lhs] + [self.variables[arg_name]
                                      for arg_name in self.argnames]
        self.miniargs = [self.b.funcarg(variable) for variable in variables]
        self.seen_return = False

    def visit_FunctionDef(self, node):
        if self.global_names:
            raise UntranslatableError("Function uses globals")

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

    def visit_Num(self, node):
        return self.b.constant(node.n)

    def generic_visit(self, node):
        raise UntranslatableError("Unsupported node of type %s" %
                                                    type(node).__name__)

class NumbaContigSpecializer(minispecializers.ContigSpecializer):
    """
    Make the contiguous specialization accept strides arguments (for generality).
    """
    def visit_FunctionNode(self, node):
        node = super(minispecializers.ContigSpecializer,
                     self).visit_FunctionNode(node)
        self.astbuilder.create_function_type(node, strides_args=True)
        return node

    def visit_StridePointer(self, node):
        self.visitchildren(node)
        return node

class MiniVectorize(object):
    """
    Vectorizer that uses minivect to produce a ufunc. The ufunc is an actual
    ufunc that dispatches to minivect for element-wise application.

    See _minidispatch.pyx for the dispatch code.
    """

    num_exprs = 0

    def __init__(self, func, fallback=basic.BasicVectorize):
        self.pyfunc = func
        self.ast = functions._get_ast(func)
        self.signatures = []
        self.fallback = fallback

    def add(self, restype, argtypes, **kwargs):
        "Add a signature for the kernel"
        # collect the signatures for compilation
        self.signatures.append((restype, argtypes, kwargs))

    def build_ufunc(self, parallel=False):
        """
        Specialize for all arrays 1D or all arrays 2D.
        Higher dimensional runtime operands need wrapping loop nests, which
        is performed by the dispatcher.
        """
        # Compile the kernels using Numba, and create a ufunc without
        # dispatcher
        dyn_ufunc, lfuncs = self.fallback_vectorize(
                minivect_dispatcher=None,
                module=minicontext.llvm_module,
                engine=minicontext.llvm_ee)

        # Try to directly map and inline the kernels by mapping to a minivect
        # AST. If this fails, we reuse the compiled kernels ('lfuncs'), and
        # generate code to call the kernels from minivect.

        # This will allow us to get parallelism at the outermost looping level,
        # as well as efficient element-wise traversal order.
        minivect_asts = []
        for lfunc, (restype, argtypes, kwargs) in zip(lfuncs, self.signatures):
            for dimensionality in (1, 2):
                # Set up types and AST mapper
                array_types = [minitypes.ArrayType(arg_type, dimensionality)
                                   for arg_type in argtypes]
                rtype = minitypes.ArrayType(restype, dimensionality)
                mapper = PythonUfunc2Minivect(decorators.context, self.pyfunc,
                                              self.ast, rtype, array_types)

                if any(array_type.is_object or array_type.is_complex
                            for array_type in array_types):
                    # Operations on complex numbers or objects are not
                    # supported by minivect, generate a kernel call
                    logging.info("Kernel not inlined, has objects/complex numbers")
                    signature = restype(*argtypes)
                    minivect_ast = build_kernel_call(
                                lfunc, signature, mapper.miniargs)
                else:
                    # Try to directly map and inline, or fall back to generating
                    # a call
                    try:
                        minivect_ast = mapper.visit(self.ast)
                    except (UntranslatableError, minierror.Error), e:
                        logging.info("Kernel not inlined: %s" % (e,))
                        minivect_ast = self.build_kernel_call(
                                    lfunc, mapper, restype, argtypes, kwargs)
                    else:
                        logging.info("Kernel directly inlined")

                minivect_asts.append((mapper, dimensionality, minivect_ast))

        # Create minivect dispatcher, and set as attribute of the dyn_ufunc
        dispatcher = self.minivect(minivect_asts, parallel)
        dispatch.set_dispatchers(dyn_ufunc, dispatcher)
        return dyn_ufunc

    def _debug(self, minifunc):
        if debug_c:
            print minicontext.debug_c(
                minifunc,
                NumbaContigSpecializer,
                # minispecializers.CTiledStridedSpecializer,
                astbuilder_cls=miniast.DynamicArgumentASTBuilder)

    def _debug_llvm(self, result):
        if debug_llvm:
            lfunc = result[-1][3][0]
            print lfunc

    def _get_function_pointers(self, mapper, codes):
        # Map minitypes to NumPy dtypes, so we can find the specialization
        # given input NumPy arrays
        result_dtype = minitypes.map_minitype_to_dtype(mapper.restype.dtype)
        dtype_args = [minitypes.map_minitype_to_dtype(arg_type)
                          for arg_type in mapper.argtypes]

        llvm_funcs = []
        function_pointers = []
        for code_result in codes:
            if code_result is None:
                llvm_funcs.append(None)
                function_pointers.append(None)
            else:
                _, _, _, (llvm_func, ctypes_func) = code_result
                llvm_funcs.append(llvm_func)
                function_pointers.append(
                    ctypes_conversion.get_pointer(minicontext, llvm_func))

        return dtype_args, function_pointers, result_dtype

    def minivect(self, asts, parallel):
        """
        Given a bunch of specialized miniasts, return a ufunc object that
        invokes the right specialization when called.
        """
        from numbapro import _minidispatch

        ufuncs = {}
        for mapper, dimensionality, ast in asts:
            minifunc = build_minifunction(ast, mapper.miniargs)
            self._debug(minifunc)
            codes = specialize(minifunc)
            dtype_args, function_pointers, result_dtype = \
                        self._get_function_pointers(mapper, codes)
            self._debug_llvm(codes)

            dtype_args.append(dimensionality)
            ufuncs[tuple(dtype_args)] = (function_pointers, result_dtype)

        return _minidispatch.MiniUFuncDispatcher(ufuncs, len(mapper.argtypes),
                                                 parallel)

    def fallback_vectorize(self, minivect_dispatcher, **kwargs):
        "Build an actual ufunc"
        return fallback_vectorize(self.fallback, self.pyfunc, self.signatures,
                                  minivect_dispatcher, **kwargs)

class ParallelMiniVectorize(MiniVectorize):

    def __init__(self, func, fallback=parallel.ParallelVectorize):
        super(ParallelMiniVectorize, self).__init__(func, fallback)

    def build_ufunc(self, parallel=True):
        assert parallel
        return super(ParallelMiniVectorize, self).build_ufunc(True)

if __name__ == '__main__':
    import time

    def vector_add(a, b):
        return a + np.sqrt(b)

    vectorizer = MiniVectorize(vector_add)
    # vectorizer = basic.BasicVectorize(vector_add)
    t = minitypes.float64
    vectorizer.add(restype=t, argtypes=[t, t])
    ufunc = vectorizer.build_ufunc(parallel=True)

    dtype = np.float64
    N = 200

    a = np.arange(N, dtype=dtype).reshape(N)
    b = a.copy()
    out = np.empty_like(a)

    ufunc(a, b, out=out)
    assert np.all(out == vector_add(a, b)), out

    t = time.time()
    for i in range(100):
        ufunc(a, b, out=out)
    print time.time() - t
