"""
This module contains the Pipeline class which provides a pluggable way to
define the transformations and the order in which they run on the AST.
"""

import logging
import functools

from numba import error
from numba import functions, naming, transforms
from numba import ast_type_inference as type_inference
from numba import ast_translate
from numba.minivect import minitypes

logger = logging.getLogger(__name__)

class Pipeline(object):
    """
    Runs a pipeline of transforms.
    """

    def __init__(self, context, func, ast, func_signature,
                 nopython=False, locals=None, order=None, codegen=False,
                 symtab=None, **kwargs):
        self.context = context
        self.func = func
        self.ast = ast
        self.func_signature = func_signature
        self.symtab = symtab
        if symtab is None:
            self.symtab = {}

        self.nopython = nopython
        self.locals = locals
        self.kwargs = kwargs

        if order is None:
            self.order = [
                'type_infer',
                'type_set',
                'transform_for',
                'specialize',
                'late_specializer',
            ]
            if codegen:
                self.order.append('codegen')
        else:
            self.order = order

    def make_specializer(self, cls, ast, **kwds):
        return cls(self.context, self.func, ast,
                   func_signature=self.func_signature, nopython=self.nopython,
                   symtab=self.symtab, **kwds)

    def insert_specializer(self, name, after):
        self.order.insert(self.order.index(after), name)

    def run_pipeline(self):
        ast = self.ast
        for method_name in self.order:
            ast = getattr(self, method_name)(ast)

        return self.func_signature, self.symtab, ast

    #
    ### Pipeline stages
    #

    def type_infer(self, ast):
        type_inferer = self.make_specializer(
                    type_inference.TypeInferer, ast, locals=self.locals)
        type_inferer.infer_types()
        self.func_signature = type_inferer.func_signature
        logger.debug("signature for %s: %s" % (self.func.func_name,
                                               self.func_signature))
        self.symtab = type_inferer.symtab
        return ast

    def type_set(self, ast):
        visitor = self.make_specializer(type_inference.TypeSettingVisitor, ast)
        visitor.visit(ast)
        return ast

    def transform_for(self, ast):
        transform = self.make_specializer(transforms.TransformForIterable, ast)
        return transform.visit(ast)

    def specialize(self, ast):
        return ast

    def late_specializer(self, ast):
        specializer = self.make_specializer(transforms.LateSpecializer, ast)
        return specializer.visit(ast)

    def codegen(self, ast):
        func_name = naming.specialized_mangle(self.func.__name__,
                                              self.func_signature.args)

        self.translator = self.make_specializer(ast_translate.LLVMCodeGenerator,
                                                ast, func_name=func_name,
                                                **self.kwargs)
        self.translator.translate()
        return ast


def run_pipeline(context, func, ast, func_signature,
                 pipeline=None, **kwargs):
    """
    Run a bunch of AST transformers and visitors on the AST.
    """
    # print __import__('ast').dump(ast)
    pipeline = pipeline or context.numba_pipeline(context, func, ast,
                                                  func_signature, **kwargs)
    return pipeline, pipeline.run_pipeline()

def _infer_types(context, func, restype=None, argtypes=None, **kwargs):
    ast = functions._get_ast(func)
    func_signature = minitypes.FunctionType(return_type=restype,
                                            args=argtypes)
    return run_pipeline(context, func, ast, func_signature, **kwargs)

def infer_types(context, func, restype=None, argtypes=None, **kwargs):
    """
    Like run_pipeline, but takes restype and argtypes instead of a FunctionType
    """
    pipeline, (sig, symtab, ast) = _infer_types(context, func, restype,
                                                argtypes, **kwargs)
    return sig, symtab, ast

def compile(context, func, restype=None, argtypes=None, ctypes=False,
            compile_only=False, **kwds):
    """
    Compile a numba annotated function.

        - decompile function into a Python ast
        - run type inference using the given input types
        - compile the function to LLVM
    """
    pipeline, (func_signature, symtab, ast) = _infer_types(
                context, func, restype, argtypes, codegen=True, **kwds)
    t = pipeline.translator

    if compile_only:
        return func_signature, t.lfunc, None

    if ctypes:
        ctypes_func = t.get_ctypes_func(kwds.get('llvm', True))
        return func_signature, t.lfunc, ctypes_func
    else:
        return func_signature, t.lfunc, t.build_wrapper_function()