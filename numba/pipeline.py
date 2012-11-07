"""
This module contains the Pipeline class which provides a pluggable way to
define the transformations and the order in which they run on the AST.
"""

import ast as ast_module
import logging
import functools
import pprint

import numba.closure
from numba import error
from numba import functions, naming, transforms, visitors
from numba import ast_type_inference as type_inference
from numba import ast_constant_folding as constant_folding
from numba import ast_translate
from numba import utils
from numba.utils import dump

from numba.minivect import minitypes

logger = logging.getLogger(__name__)

class Pipeline(object):
    """
    Runs a pipeline of transforms.
    """

    order = [
        'const_folding',
        'type_infer',
        'type_set',
        'closure_type_inference',
        'transform_for',
        'specialize',
        'late_specializer',
        'fix_ast_locations',
        'codegen',
    ]

    mixins = {}
    _current_pipeline_stage = None

    def __init__(self, context, func, ast, func_signature,
                 nopython=False, locals=None, order=None, codegen=False,
                 symtab=None, **kwargs):
        self.context = context
        self.func = func
        self.ast = ast
        self.func_signature = func_signature
        ast.pipeline = self

        func_name = kwargs.get('name')
        self.func_name = func_name or naming.specialized_mangle(
                           self.func.__name__, self.func_signature.args)

        self.symtab = symtab
        if symtab is None:
            self.symtab = {}

        self.llvm_module = kwargs.get('llvm_module', None)
        if self.llvm_module is None:
            self.llvm_module = self.context.function_cache.module

        self.nopython = nopython
        self.locals = locals
        self.kwargs = kwargs

        if order is None:
            self.order = list(Pipeline.order)
            if not codegen:
                self.order.remove('codegen')
        else:
            self.order = order

    def make_specializer(self, cls, ast, **kwds):
        "Create a visitor or transform and add any mixins"
        if self._current_pipeline_stage in self.mixins:
            before, after = self.mixins[self._current_pipeline_stage]
            classes = tuple(before + [cls] + after)
            name = '__'.join(cls.__name__ for cls in classes)
            cls = type(name, classes, {})

        return cls(self.context, self.func, ast,
                   func_signature=self.func_signature, nopython=self.nopython,
                   symtab=self.symtab, **kwds)

    def insert_specializer(self, name, after):
        "Insert a new transform or visitor into the pipeline"
        self.order.insert(self.order.index(after) + 1, name)

    def try_insert_specializer(self, name, after):
        if after in self.order:
            self.insert_specializer(name, after)

    @classmethod
    def add_mixin(cls, pipeline_stage, transform, before=False):
        before_mixins, after_mixins = cls.mixins.get(pipeline_stage, ([], []))
        if before:
            before_mixins.append(transform)
        else:
            after_mixins.append(transform)

        cls.mixins[pipeline_stage] = before_mixins, after_mixins

    def run_pipeline(self):
        ast = self.ast
        for method_name in self.order:
            if __debug__ and logger.getEffectiveLevel() < logging.DEBUG:
                stage_tuple = (method_name, utils.ast2tree(ast))
                logger.debug(pprint.pformat(stage_tuple))

            self._current_pipeline_stage = method_name
            ast = getattr(self, method_name)(ast)

        return self.func_signature, self.symtab, ast

    #
    ### Pipeline stages
    #
    
    def const_folding(self, ast):
        const_marker = self.make_specializer(constant_folding.ConstantMarker,
                                             ast)
        const_marker.visit(ast)
        constvars = const_marker.get_constants()
        const_folder = self.make_specializer(constant_folding.ConstantFolder,
                                             ast, constvars=constvars)
        return const_folder.visit(ast)

    def type_infer(self, ast):
        type_inferer = self.make_specializer(
                    type_inference.TypeInferer, ast, locals=self.locals,
                    **self.kwargs)
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

    def closure_type_inference(self, ast):
        type_inferer = self.make_specializer(
                            numba.closure.ClosureTypeInferer, ast)
        return type_inferer.visit(ast)

    def transform_for(self, ast):
        transform = self.make_specializer(transforms.TransformForIterable, ast)
        return transform.visit(ast)

    def specialize(self, ast):
        return ast

    def late_specializer(self, ast):
        specializer = self.make_specializer(transforms.LateSpecializer, ast)
        return specializer.visit(ast)

    def fix_ast_locations(self, ast):
        fixer = self.make_specializer(FixMissingLocations, ast)
        fixer.visit(ast)
        return ast

    def codegen(self, ast):
        self.translator = self.make_specializer(ast_translate.LLVMCodeGenerator,
                                                ast, func_name=self.func_name,
                                                **self.kwargs)
        self.translator.translate()
        return ast


class FixMissingLocations(visitors.NumbaVisitor):

    def __init__(self, context, func, ast, func_signature=None, nopython=0,
                 symtab=None):
        super(FixMissingLocations, self).__init__(context, func, ast,
                                                  func_signature, nopython,
                                                  symtab)
        self.lineno = getattr(ast, 'lineno', 1)
        self.col_offset = getattr(ast, 'col_offset', 0)

    def visit(self, node):
        if not hasattr(node, 'lineno'):
            node.lineno = self.lineno
            node.col_offset = self.col_offset

        super(FixMissingLocations, self).visit(node)

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
                                                argtypes, order=['type_infer'],
                                                **kwargs)
    return sig, symtab, ast

def infer_types_from_ast_and_sig(context, dummy_func, ast, signature, **kwargs):
    return run_pipeline(context, dummy_func, ast, signature,
                        order=['type_infer'], **kwargs)

def get_wrapper(translator, ctypes=False):
    if ctypes:
        return translator.get_ctypes_func()
    else:
        return translator.build_wrapper_function()

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
        return func_signature, t, None

    return func_signature, t, get_wrapper(t, ctypes)

def compile_from_sig(context, func, signature, **kwds):
    return compile(context, func, signature.return_type, signature.args,
                   **kwds)