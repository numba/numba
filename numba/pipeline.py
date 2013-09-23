# -*- coding: utf-8 -*-
"""
This module contains the Pipeline class which provides a pluggable way to
define the transformations and the order in which they run on the AST.
"""
from __future__ import print_function, division, absolute_import

import os
import sys
import ast as ast_module
import logging
import pprint
import random
import types
import copy
import llvm.core as lc

# import numba.closures
from numba import PY3
from numba import error
from numba import functions
from numba import transforms
from numba import control_flow
from numba import closures
from numba import reporting
from numba import normalize
from numba import validate
from numba.array_validation import ArrayValidator
from numba.viz import cfgviz
from numba import typesystem
from numba.codegen import llvmwrapper
from numba import ast_constant_folding as constant_folding
from numba.control_flow import ssa, cfstats
from numba.codegen import translate
from numba import utils
from numba.missing import FixMissingLocations
from numba.type_inference import infer as type_inference
from numba.asdl import schema
from numba.prettyprint import (dump_ast, dump_cfg, dump_annotations,
                               dump_llvm, dump_optimized)
import numba.visitors

from numba.specialize import comparisons
from numba.specialize import loops
from numba.specialize import exceptions
from numba.specialize import funccalls
from numba.specialize import exttypes

from numba import astsix

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

def get_locals(ast, locals_dict):
    # TODO: Remove this
    if locals_dict is None:
        locals_dict = getattr(ast, "locals_dict", {})
    elif hasattr(ast, "locals_dict"):
        assert ast.locals_dict is locals_dict

    ast.locals_dict = locals_dict
    return locals_dict

def module_name(func):
    if func is None:
        name = "NoneFunc"
        func_id = random.randrange(1000000)
    else:
        name = '%s.%s' % (func.__module__, func.__name__)
        func_id = id(func)

    return 'tmp.module.%s.%x' % (name, func_id)

#------------------------------------------------------------------------
# Entry points
#------------------------------------------------------------------------

def run_pipeline2(env, func, func_ast, func_signature,
                  pipeline=None, **kwargs):
    assert pipeline is None
    assert kwargs.get('order', None) is None
    logger.debug(pprint.pformat(kwargs))
    kwargs['llvm_module'] = lc.Module.new(module_name(func))
    with env.TranslationContext(env, func, func_ast, func_signature,
                                **kwargs) as func_env:
        pipeline = env.get_pipeline(kwargs.get('pipeline_name', None))
        post_ast = pipeline(func_ast, env)
        func_signature = func_env.func_signature
        symtab = func_env.symtab

    return func_env, (func_signature, symtab, post_ast)

def run_env(env, func_env, **kwargs):
    env.translation.push_env(func_env)
    pipeline = env.get_pipeline(kwargs.get('pipeline_name', None))
    try:
        pipeline(func_env.ast, env)
    finally:
        env.translation.pop()

def _infer_types2(env, func, restype=None, argtypes=None, **kwargs):
    ast = functions._get_ast(func)
    func_signature = typesystem.function(restype, argtypes)
    return run_pipeline2(env, func, ast, func_signature, **kwargs)

def infer_types2(env, func, restype=None, argtypes=None, **kwargs):
    """
    Like run_pipeline, but takes restype and argtypes instead of a function
    """
    pipeline, (sig, symtab, ast) = _infer_types2(
        env, func, restype, argtypes, pipeline_name='type_infer', **kwargs)
    return sig, symtab, ast


def compile2(env, func, restype=None, argtypes=None, ctypes=False,
             compile_only=False, func_ast=None, **kwds):
    """
    Compile a numba annotated function.

        - decompile function into a Python ast
        - run type inference using the given input types
        - compile the function to LLVM
    """
    # Let the pipeline create a module for the function it is compiling
    # and the user will link that in.
    assert 'llvm_module' not in kwds
    kwds['llvm_module'] = lc.Module.new(module_name(func))
    logger.debug(kwds)
    if func_ast is None:
        func_ast = functions._get_ast(func)
    else:
        func_ast = copy.deepcopy(func_ast)
    func_signature = typesystem.function(restype, argtypes)
    #pipeline, (func_signature, symtab, ast) = _infer_types2(
    #            env, func, restype, argtypes, codegen=True, **kwds)
    with env.TranslationContext(env, func, func_ast, func_signature,
                                need_lfunc_wrapper=not compile_only,
                                **kwds) as func_env:
        pipeline = env.get_pipeline(kwds.get('pipeline_name', None))
        func_ast.pipeline = pipeline
        post_ast = pipeline(func_ast, env)
        func_signature = func_env.func_signature
        symtab = func_env.symtab
        t = func_env.translator
    return func_env


#------------------------------------------------------------------------
# Pipeline refactored code
#------------------------------------------------------------------------

class PipelineStage(object):

    is_composed = False

    def check_preconditions(self, ast, env):
        return True

    def check_postconditions(self, ast, env):
        return True

    def transform(self, ast, env):
        raise NotImplementedError('%r does not implement transform!' %
                                  type(self))

    def make_specializer(self, cls, ast, env, **kws):
        crnt = env.translation.crnt
        kws = kws.copy()
        kws.update(func_signature=crnt.func_signature,
                   nopython=env.translation.nopython,
                   symtab=crnt.symtab,
                   func_name=crnt.func_name,
                   llvm_module=crnt.llvm_module,
                   func_globals=crnt.function_globals,
                   locals=crnt.locals,
                   allow_rebind_args=env.translation.allow_rebind_args,
                   warn=env.translation.crnt.warn,
                   is_closure=crnt.is_closure,
                   closures=crnt.closures,
                   closure_scope=crnt.closure_scope,
                   env=env)
        return cls(env.context, crnt.func, ast, **kws)

    def __call__(self, ast, env):
        if env.stage_checks: self.check_preconditions(ast, env)

        if self.is_composed:
            ast = self.transform(ast, env)
        else:
            try:
                ast = self.transform(ast, env)
            except error.NumbaError as e:
                func_env = env.translation.crnt
                error_env = func_env.error_env
                if func_env.is_closure:
                    flags, parent_func_env = env.translation.stack[-2]
                    error_env.merge_in(parent_func_env.error_env)
                elif not e.has_report:
                    reporting.report(env, exc=e)
                raise

        env.translation.crnt.ast = ast
        if env.stage_checks: self.check_postconditions(ast, env)
        return ast

class SimplePipelineStage(PipelineStage):

    transformer = None

    def transform(self, ast, env):
        transform = self.make_specializer(self.transformer, ast, env)
        return transform.visit(ast)


class AST3to2(PipelineStage):

    def transform(self, ast, env):
        if not PY3:
            return ast
        return astsix.AST3to2().visit(ast)


def ast3to2(ast, env):
    if not PY3:
        return ast
    return astsix.AST3to2().visit(ast)


def resolve_templates(ast, env):
    # TODO: Unify with decorators module
    crnt = env.translation.crnt
    if crnt.template_signature is not None:
        from numba import typesystem

        argnames = [name.id for name in ast.args.args]
        argtypes = list(crnt.func_signature.args)

        template_context, signature = typesystem.resolve_templates(
            crnt.locals, crnt.template_signature, argnames, argtypes)
        crnt.func_signature = signature

    return ast


def validate_signature(tree, env):
    arg_types = env.translation.crnt.func_signature.args
    if (isinstance(tree, ast_module.FunctionDef) and
        len(arg_types) != len(tree.args.args)):
        raise error.NumbaError(
            "Incorrect number of types specified in @jit() for function %r" %
            env.crnt.func_name)

    return tree

def validate_arrays(ast, env):
    ArrayValidator(env).visit(ast)
    return ast

def update_signature(tree, env):
    func_env = env.translation.crnt
    func_signature = func_env.func_signature

    restype = func_signature.return_type
    if restype and (restype.is_struct or restype.is_complex or restype.is_datetime or restype.is_timedelta):
        # Change signatures returning complex numbers or structs to
        # signatures taking a pointer argument to a complex number
        # or struct
        func_signature = func_signature.return_type(*func_signature.args)
        func_env.func_signature = func_signature

    return tree


def get_lfunc(env, func_env):
    lfunc = func_env.llvm_module.add_function(
        func_env.func_signature.to_llvm(env.context),
        func_env.mangled_name)
    return lfunc


def create_lfunc(tree, env):
    """
    Update the FunctionEnvironment with an LLVM function if the signature
    is known (try this before type inference to support recursion).
    """
    func_env = env.translation.crnt

    if (not func_env.lfunc and func_env.func_signature and
            func_env.func_signature.return_type):
        assert func_env.llvm_module is not None
        lfunc = get_lfunc(env, func_env)

        func_env.lfunc = lfunc
        if func_env.func:
            env.specializations.register_specialization(func_env)

    return tree

def create_lfunc1(tree, env):
    func_env = env.translation.crnt
    if not func_env.is_closure:
        create_lfunc(tree, env)

    return tree

def create_lfunc2(tree, env):
    func_env = env.translation.crnt
    assert func_env.func_signature and func_env.func_signature.return_type
    return create_lfunc1(tree, env)

def create_lfunc3(tree, env):
    func_env = env.translation.crnt
    create_lfunc(tree, env)
    return tree

# ______________________________________________________________________

class ValidateASTStage(PipelineStage):
    def transform(self, ast, env):
        validate.ValidateAST().visit(ast)
        return ast

class NormalizeASTStage(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(normalize.NormalizeAST, ast, env)
        return transform.visit(ast)

# ______________________________________________________________________

class ControlFlowAnalysis(PipelineStage):
    _pre_condition_schema = None

    @property
    def pre_condition_schema(self):
        if self._pre_condition_schema is None:
            self._pre_condition_schema = schema.load('Python.asdl')
        return self._pre_condition_schema

    def check_preconditions(self, ast, env):
        self.pre_condition_schema.verify(ast)  # raises exception on error
        return True

    def transform(self, ast, env):
        transform = self.make_specializer(control_flow.ControlFlowAnalysis,
                                          ast, env)
        ast = transform.visit(ast)
        env.translation.crnt.symtab = transform.symtab
        ast.flow = transform.flow
        return ast


class ConstFolding(PipelineStage):
    def check_preconditions(self, ast, env):
        assert not hasattr(env.crnt, 'constvars')
        return super(ConstFolding, self).check_preconditions(ast, env)

    def check_postconditions(self, ast, env):
        assert hasattr(env.crnt, 'constvars')
        return super(ConstFolding, self).check_postconditions(ast, env)

    def transform(self, ast, env):
        const_marker = self.make_specializer(constant_folding.ConstantMarker,
                                             ast, env)
        const_marker.visit(ast)
        constvars = const_marker.get_constants()
        # FIXME: Make constvars a property of the FunctionEnvironment,
        # or nix this transformation pass.
        env.translation.crnt.constvars = constvars
        const_folder = self.make_specializer(constant_folding.ConstantFolder,
                                             ast, env, constvars=constvars)
        return const_folder.visit(ast)


class TypeInfer(PipelineStage):
    def check_preconditions(self, ast, env):
        assert env.translation.crnt.symtab is not None
        return super(TypeInfer, self).check_preconditions(ast, env)

    def transform(self, ast, env):
        crnt = env.translation.crnt
        type_inferer = self.make_specializer(type_inference.TypeInferer,
                                             ast, env, **crnt.kwargs)
        type_inferer.infer_types()
        crnt.func_signature = type_inferer.func_signature
        logger.debug("signature for %s: %s", crnt.func_name,
                     crnt.func_signature)
        crnt.symtab = type_inferer.symtab
        return ast


class TypeSet(PipelineStage):
    def transform(self, ast, env):
        visitor = self.make_specializer(type_inference.TypeSettingVisitor, ast,
                                        env)
        visitor.visit(ast)
        return ast


class ClosureTypeInference(PipelineStage):
    def transform(self, ast, env):
        type_inferer = self.make_specializer(
                            numba.closures.ClosureTypeInferer, ast, env)
        return type_inferer.visit(ast)


class TransformFor(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(loops.TransformForIterable, ast,
                                          env)
        return transform.visit(ast)

class TransformBuiltinLoops(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(loops.TransformBuiltinLoops, ast,
                                          env)
        return transform.visit(ast)

#----------------------------------------------------------------------------
# Prange
#----------------------------------------------------------------------------

def run_prange(name):
    def wrapper(ast, env):
        from numba import parallel
        stage = getattr(parallel, name)
        return PipelineStage().make_specializer(stage, ast, env).visit(ast)

    wrapper.__name__ = name
    return wrapper

ExpandPrange = run_prange('PrangeExpander')
RewritePrangePrivates = run_prange('PrangePrivatesReplacer')
CleanupPrange = run_prange('PrangeCleanup')


class UpdateAttributeStatements(PipelineStage):
    def transform(self, ast, env):
        func_env = env.translation.crnt

        for block in func_env.flow.blocks:
            stats = []
            for cf_stat in block.stats:
                if (isinstance(cf_stat, cfstats.AttributeAssignment) and
                        isinstance(cf_stat.lhs, ast_module.Attribute)):
                    value = cf_stat.lhs.value
                    if (isinstance(value, ast_module.Name) and
                                value.id in func_env.kill_attribute_assignments):
                        cf_stat = None

                if cf_stat:
                    stats.append(cf_stat)

            block.stats = stats

        return ast

#----------------------------------------------------------------------------
# Specializing/Lowering Transforms
#----------------------------------------------------------------------------

class Specialize(PipelineStage):
    def transform(self, ast, env):
        return ast

class RewriteArrayExpressions(PipelineStage):
    def transform(self, ast, env):
        from numba import array_expressions

        transformer = self.make_specializer(
            array_expressions.ArrayExpressionRewriteNative, ast, env)
        return transformer.visit(ast)

class SpecializeComparisons(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(comparisons.SpecializeComparisons,
                                          ast, env)
        return transform.visit(ast)


class SpecializeSSA(PipelineStage):
    def transform(self, ast, env):
        ssa.specialize_ssa(ast)
        return ast

class SpecializeClosures(SimplePipelineStage):
    transformer = closures.ClosureSpecializer

class Optimize(PipelineStage):
    def transform(self, ast, env):
        return ast

class SpecializeLoops(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(loops.SpecializeObjectIteration, ast,
                                          env)
        return transform.visit(ast)

class LowerRaise(PipelineStage):
    def transform(self, ast, env):
        return self.make_specializer(exceptions.LowerRaise, ast, env).visit(ast)

class LateSpecializer(PipelineStage):
    def transform(self, ast, env):
        specializer = self.make_specializer(transforms.LateSpecializer, ast,
                                            env)
        return specializer.visit(ast)

class ExtensionTypeLowerer(PipelineStage):
    def transform(self, ast, env):
        specializer = self.make_specializer(exttypes.ExtensionTypeLowerer,
                                            ast, env)
        return specializer.visit(ast)

class SpecializeFunccalls(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(funccalls.FunctionCallSpecializer,
                                          ast, env)
        return transform.visit(ast)


class SpecializeExceptions(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(exceptions.ExceptionSpecializer, ast,
                                          env)
        return transform.visit(ast)


def cleanup_symtab(ast, env):
    "Pop original variables from the symtab"
    for var in env.translation.crnt.symtab.values():
        if not var.parent_var and var.renameable:
            env.translation.crnt.symtab.pop(var.name, None)
    return ast


class FixASTLocations(PipelineStage):
    def transform(self, ast, env):
        lineno = getattr(ast, 'lineno', 1)
        col_offset = getattr(ast, 'col_offset', 1)
        FixMissingLocations(lineno, col_offset).visit(ast)
        return ast

class CodeGen(PipelineStage):
    def transform(self, ast, env):
        func_env = env.translation.crnt
        func_env.translator = self.make_specializer(
            translate.LLVMCodeGenerator, ast, env,
            **func_env.kwargs)

        func_env.translator.translate()
        func_env.lfunc = func_env.translator.lfunc
        return ast

class PostPass(PipelineStage):
    def transform(self, ast, env):
        for postpass_name, postpass in env.crnt.postpasses.iteritems():
            env.crnt.lfunc = postpass(env,
                                      env.llvm_context.execution_engine,
                                      env.crnt.llvm_module,
                                      env.crnt.lfunc)
        return ast

class LinkingStage(PipelineStage):
    """
    Link the resulting LLVM function into the global fat module.
    """

    def transform(self, ast, env):
        func_env = env.translation.crnt

        # Link libraries into module
        env.context.intrinsic_library.link(func_env.lfunc.module)
        # env.context.cbuilder_library.link(func_env.lfunc.module)
        env.constants_manager.link(func_env.lfunc.module)

        lfunc_pointer = 0
        if func_env.link:
            # Link function into fat LLVM module
            func_env.lfunc = env.llvm_context.link(func_env.lfunc)
            func_env.translator.lfunc = func_env.lfunc
            lfunc_pointer = func_env.translator.lfunc_pointer

        func_env.lfunc_pointer = lfunc_pointer

        return ast

class WrapperStage(PipelineStage):
    """
    Build a wrapper LLVM function around the compiled numba function to call
    it from Python.
    """

    def transform(self, ast, env):
        func_env = env.translation.crnt
        if func_env.is_closure:
            wrap = func_env.need_closure_wrapper
        else:
            wrap = func_env.wrap

        if wrap:
            numbawrapper, lfuncwrapper, _ = (
                llvmwrapper.build_wrapper_function(env))
            func_env.numba_wrapper_func = numbawrapper
            func_env.llvm_wrapper_func = lfuncwrapper

            # Set pointer to function for external code and numba.addressof()
            numbawrapper.lfunc_pointer = func_env.lfunc_pointer

        return ast

class ErrorReporting(PipelineStage):
    "Sort and issue warnings and errors"
    def transform(self, ast, env):
        reporting.report(env)
        return ast

class ComposedPipelineStage(PipelineStage):

    is_composed = True

    def __init__(self, stages=None):
        if stages is None:
            stages = []
        self.stages = [self.check_stage(stage)[1] for stage in stages]

    @staticmethod
    def check_stage(stage):
        def _check_stage_object(stage_obj):
            if (isinstance(stage_obj, type) and
                    issubclass(stage_obj, PipelineStage)):
                stage_obj = stage_obj()
            return stage_obj

        if isinstance(stage, str):
            name = stage
            def _stage(ast, env):
                stage_obj = getattr(env.pipeline_stages, name)
                return _check_stage_object(stage_obj)(ast, env)
            _stage.__name__ = name
            stage = _stage
        else:
            name = stage.__name__
            stage = _check_stage_object(stage)

        return name, stage

    def transform(self, ast, env):
        logger.debug('Running composed stages: %s', self.stages)
        for stage in self.stages:
            if env.debug:
                stage_tuple = (stage, utils.ast2tree(ast))
                logger.debug(pprint.pformat(stage_tuple))
            ast = stage(ast, env)
        return ast

    @classmethod
    def compose(cls, stage0, stage1):
        if isinstance(stage0, ComposedPipelineStage):
            stage0s = stage0.stages
        else:
            stage0s = [check_stage(stage0)[1]]
        if isinstance(stage1, ComposedPipelineStage):
            stage1s = stage1.stages
        else:
            stage1s = [check_stage(stage1)[1]]
        return cls(stage0s + stage1s)
