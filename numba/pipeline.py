"""
This module contains the Pipeline class which provides a pluggable way to
define the transformations and the order in which they run on the AST.
"""

import inspect
import ast as ast_module
import logging
import pprint
import random
from timeit import default_timer as _timer

import llvm.core as lc

# import numba.closures
from numba import PY3
from numba import error
from numba import functions
from numba import naming
from numba import transforms
from numba import control_flow
from numba import optimize
from numba import closures
from numba import reporting
from numba import ast_constant_folding as constant_folding
from numba.control_flow import ssa
from numba.codegen import translate
from numba import utils
from numba.type_inference import infer as type_inference
from numba.utils import dump, TypedProperty
from numba.asdl import schema
from numba.minivect import minitypes
import numba.visitors
from numba.specialize import comparisons, loops, exceptions, funccalls

logger = logging.getLogger(__name__)

def get_locals(ast, locals_dict):
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


class Pipeline(object):
    """
    Runs a pipeline of transforms.
    """

    order = [
        'resolve_templates',
        'validate_signature',
        'cfg',
        #'dump_cfg',
        #'const_folding',
        'type_infer',
        'type_set',
        'dump_cfg',
        'closure_type_inference',
        'transform_for',
        'specialize',
        'specialize_comparisons',
        'specialize_ssa',
        'specialize_closures',
        'optimize',
        'preloader',
        'specialize_loops',
        'late_specializer',
        'specialize_funccalls',
        'specialize_exceptions',
        'fix_ast_locations',
        'cleanup_symtab',
        'codegen',
    ]

    mixins = {}
    _current_pipeline_stage = None

    def __init__(self, context, func, ast, func_signature,
                 nopython=False, locals=None, order=None, codegen=False,
                 symtab=None, allow_rebind_args=True, template_signature=None,
                 is_closure=False, **kwargs):
        self.context = context
        self.func = func
        self.ast = ast
        self.func_signature = func_signature
        self.template_signature = template_signature

        # Whether argument variables may be rebound to different types.
        # e.g. def f(a): a = "hello" ;; f(0.0)
        self.allow_rebind_args = allow_rebind_args

        ast.pipeline = self

        assert "name" not in kwargs
        self.mangled_name = kwargs.get('mangled_name')

        self.symtab = symtab
        if symtab is None:
            self.symtab = {}

        # Let the pipeline create a module for the function it is compiling
        # and the user will link that in.
        assert 'llvm_module' not in kwargs
        self.llvm_module = lc.Module.new(module_name(func))

        self.nopython = nopython
        self.locals = get_locals(ast, locals)
        self.is_closure = is_closure

        self.closures = kwargs.pop("closures", {})
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

        kwds.setdefault("mangled_name", self.mangled_name)

        assert 'llvm_module' not in kwds
        return cls(self.context, self.func, ast,
                   func_signature=self.func_signature, nopython=self.nopython,
                   symtab=self.symtab,
                   llvm_module=self.llvm_module,
                   locals=self.locals,
                   allow_rebind_args=self.allow_rebind_args,
                   closures=self.closures,
                   is_closure=self.is_closure,
                   **kwds)

    def insert_specializer(self, name, after=None, before=None):
        "Insert a new transform or visitor into the pipeline"
        if after:
            index = self.order.index(after) + 1
        else:
            index = self.order.index(before)

        self.order.insert(index, name)

    def try_insert_specializer(self, name, after=None, before=None):
        if after and after in self.order:
            self.insert_specializer(name, after=after)
        if before and before in self.order:
            self.insert_specializer(name, before=before)

    @classmethod
    def add_mixin(cls, pipeline_stage, transform, before=False):
        before_mixins, after_mixins = cls.mixins.get(pipeline_stage, ([], []))
        if before:
            before_mixins.append(transform)
        else:
            after_mixins.append(transform)

        cls.mixins[pipeline_stage] = before_mixins, after_mixins

    def run_pipeline(self):
        # Uses a special logger for logging profiling information.
        logger = logging.getLogger("numba.pipeline.profiler")
        ast = self.ast
        talpha = _timer() # for profiling complete pipeline
        for method_name in self.order:
            ts = _timer() # for profiling individual stage
            if __debug__ and logger.getEffectiveLevel() < logging.DEBUG:
                stage_tuple = (method_name, utils.ast2tree(ast))
                logger.debug(pprint.pformat(stage_tuple))

            self._current_pipeline_stage = method_name
            ast = getattr(self, method_name)(ast)

            if __debug__ and logger.getEffectiveLevel() < logging.DEBUG:
                te = _timer() #  for profiling individual stage
                logger.debug("%X pipeline stage %30s:\t%.3fms",
                            id(self), method_name, (te - ts) * 1000)

        tomega = _timer() # for profiling complete pipeline
        logger.debug("%X pipeline entire:\t\t\t\t\t%.3fms",
                    id(self), (tomega - talpha) * 1000)

        return self.func_signature, self.symtab, ast

    #
    ### Pipeline stages
    #

    def resolve_templates(self, ast):
        # TODO: Unify with decorators module
        if self.template_signature is not None:
            from numba import typesystem

            if PY3:
                argnames = [name.arg for name in ast.args.args]
            else:
                argnames = [name.id for name in ast.args.args]

            argtypes = list(self.func_signature.args)

            typesystem.resolve_templates(self.locals, self.template_signature,
                                         argnames, argtypes)
            self.func_signature = minitypes.FunctionType(
                    return_type=self.func_signature.return_type,
                    args=tuple(argtypes))

        return ast

    def validate_signature(self, tree):
        arg_types = self.func_signature.args
        if (isinstance(tree, ast_module.FunctionDef) and
                len(arg_types) != len(tree.args.args)):
            raise error.NumbaError(
                "Incorrect number of types specified in @jit()")

        return tree

    def cfg(self, ast):
        transform = self.make_specializer(
                control_flow.ControlFlowAnalysis, ast, **self.kwargs)
        ast = transform.visit(ast)
        self.symtab = transform.symtab
        ast.flow = transform.flow
        self.ast.cfg_transform = transform
        return ast

    def dump_cfg(self, ast):
        if self.ast.cfg_transform.graphviz:
            self.ast.cfg_transform.render_gv(ast)
        return ast

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
                type_inference.TypeInferer, ast, **self.kwargs)
        type_inferer.infer_types()

        self.func_signature = type_inferer.func_signature
        logger.debug("signature for %s: %s", self.mangled_name,
                     self.func_signature)
        self.symtab = type_inferer.symtab
        return ast

    def type_set(self, ast):
        visitor = self.make_specializer(type_inference.TypeSettingVisitor, ast)
        visitor.visit(ast)
        return ast

    def closure_type_inference(self, ast):
        type_inferer = self.make_specializer(
                            closures.ClosureTypeInferer, ast,
                            warn=self.kwargs.get("warn", True))
        return type_inferer.visit(ast)

    def transform_for(self, ast):
        transform = self.make_specializer(loops.TransformForIterable, ast)
        return transform.visit(ast)

    def specialize(self, ast):
        return ast

    def specialize_comparisons(self, ast):
        transform = self.make_specializer(comparisons.SpecializeComparisons, ast)
        return transform.visit(ast)

    def specialize_ssa(self, ast):
        ssa.specialize_ssa(ast)
        return ast

    def specialize_closures(self, ast):
        transform = self.make_specializer(closures.ClosureSpecializer, ast)
        return transform.visit(ast)

    def specialize_loops(self, ast):
        transform = self.make_specializer(loops.SpecializeObjectIteration, ast)
        return transform.visit(ast)

    def specialize_funccalls(self, ast):
        transform = self.make_specializer(funccalls.FunctionCallSpecializer, ast)
        return transform.visit(ast)

    def specialize_exceptions(self, ast):
        transform = self.make_specializer(exceptions.ExceptionSpecializer, ast)
        return transform.visit(ast)

    def optimize(self, ast):
        return ast

    def preloader(self, ast):
        return self.make_specializer(optimize.Preloader, ast).visit(ast)

    def late_specializer(self, ast):
        specializer = self.make_specializer(transforms.LateSpecializer, ast)
        return specializer.visit(ast)

    def fix_ast_locations(self, ast):
        fixer = self.make_specializer(FixMissingLocations, ast)
        fixer.visit(ast)
        return ast

    def cleanup_symtab(self, ast):
        "Pop original variables from the symtab"
        for var in ast.symtab.values():
            if not var.parent_var and var.renameable:
                ast.symtab.pop(var.name, None)

        return ast

    def codegen(self, ast):
        self.translator = self.make_specializer(translate.LLVMCodeGenerator,
                                                ast, **self.kwargs)
        self.translator.translate()
        return ast


class FixMissingLocations(numba.visitors.NumbaVisitor):

    def __init__(self, context, func, ast, *args, **kwargs):
        super(FixMissingLocations, self).__init__(context, func, ast,
                                                  *args, **kwargs)
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

def run_pipeline2(env, func, func_ast, func_signature,
                  pipeline=None, **kwargs):
    assert pipeline is None
    assert kwargs.get('order', None) is None
    logger.debug(pprint.pformat(kwargs))
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

def _infer_types(context, func, restype=None, argtypes=None, **kwargs):
    ast = functions._get_ast(func)
    func_signature = minitypes.FunctionType(return_type=restype,
                                            args=argtypes)
    return run_pipeline(context, func, ast, func_signature, **kwargs)

def _infer_types2(env, func, restype=None, argtypes=None, **kwargs):
    ast = functions._get_ast(func)
    func_signature = minitypes.FunctionType(return_type=restype,
                                            args=argtypes)
    return run_pipeline2(env, func, ast, func_signature, **kwargs)

def infer_types(context, func, restype=None, argtypes=None, **kwargs):
    """
    Like run_pipeline, but takes restype and argtypes instead of a FunctionType
    """
    pipeline, (sig, symtab, ast) = _infer_types(context, func, restype,
                                                argtypes, order=['cfg', 'type_infer'],
                                                **kwargs)
    return sig, symtab, ast

def infer_types2(env, func, restype=None, argtypes=None, **kwargs):
    """
    Like run_pipeline, but takes restype and argtypes instead of a FunctionType
    """
    pipeline, (sig, symtab, ast) = _infer_types2(
        env, func, restype, argtypes, pipeline_name='type_infer', **kwargs)
    return sig, symtab, ast

def infer_types_from_ast_and_sig(context, dummy_func, ast, signature, **kwargs):
    return run_pipeline(context, dummy_func, ast, signature,
                        order=['cfg', 'type_infer'], **kwargs)

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
    assert 'llvm_module' not in kwds
    pipeline, (func_signature, symtab, ast) = _infer_types(
                context, func, restype, argtypes, codegen=True, **kwds)
    t = pipeline.translator

    if compile_only:
        return func_signature, t, None

    # link intrinsic library
    context.intrinsic_library.link(t.lfunc.module)

    # link into the JIT module
    t.link()
    return func_signature, t, get_wrapper(t, ctypes)

def compile2(env, func, restype=None, argtypes=None, ctypes=False,
             compile_only=False, **kwds):
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
    func_ast = functions._get_ast(func)
    func_signature = minitypes.FunctionType(return_type=restype,
                                            args=argtypes)
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

def compile_from_sig(context, func, signature, **kwds):
    return compile(context, func, signature.return_type, signature.args,
                   **kwds)

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
                else:
                    reporting.report(error_env, error_env.enable_post_mortem,
                                     exc=e)
                raise

        env.translation.crnt.ast = ast
        if env.stage_checks: self.check_postconditions(ast, env)
        return ast

class SimplePipelineStage(PipelineStage):

    transformer = None

    def transform(self, ast, env):
        transform = self.make_specializer(self.transformer, ast, env)
        return transform.visit(ast)

def resolve_templates(ast, env):
    # TODO: Unify with decorators module
    crnt = env.translation.crnt
    if crnt.template_signature is not None:
        from numba import typesystem

        argnames = [arg.id for arg in ast.args.args]
        argtypes = list(crnt.func_signature.args)

        typesystem.resolve_templates(crnt.locals, crnt.template_signature,
                                     argnames, argtypes)
        crnt.func_signature = minitypes.FunctionType(
            return_type=crnt.func_signature.return_type,
            args=tuple(argtypes))

    return ast


def validate_signature(tree, env):
    arg_types = env.translation.crnt.func_signature.args
    if (isinstance(tree, ast_module.FunctionDef) and
        len(arg_types) != len(tree.args.args)):
        raise error.NumbaError(
            "Incorrect number of types specified in @jit()")

    return tree

def update_signature(tree, env):
    func_env = env.translation.crnt
    func_signature = func_env.func_signature

    restype = func_signature.return_type
    if restype and (restype.is_struct or restype.is_complex):
        # Change signatures returning complex numbers or structs to
        # signatures taking a pointer argument to a complex number
        # or struct
        func_signature = func_signature.return_type(*func_signature.args)
        func_signature.struct_by_reference = True
        func_env.func_signature = func_signature

    return tree

def create_lfunc(tree, env):
    """
    Update the FunctionEnvironment with an LLVM function if the signature
    is known (try this before type inference to support recursion).
    """
    func_env = env.translation.crnt

    if (not func_env.lfunc and func_env.func_signature and
            func_env.func_signature.return_type):
        assert func_env.llvm_module is not None
        lfunc = func_env.llvm_module.add_function(
                func_env.func_signature.to_llvm(env.context),
                func_env.mangled_name)

        func_env.lfunc = lfunc
        if func_env.func:
            env.specializations.register_specialization(func_env)

    return tree

def create_lfunc2(tree, env):
    func_env = env.translation.crnt
    assert func_env.func_signature and func_env.func_signature.return_type
    return create_lfunc(tree, env)

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
        env.translation.crnt.ast.cfg_transform = transform
        return ast


def dump_cfg(ast, env):
    if env.translation.crnt.cfg_transform.graphviz:
        env.translation.crnt.cfg_transform.render_gv(ast)
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


class Preloader(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(optimize.Preloader, ast, env)
        return transform.visit(ast)


class SpecializeLoops(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(loops.SpecializeObjectIteration, ast,
                                          env)
        return transform.visit(ast)


class LateSpecializer(PipelineStage):
    def transform(self, ast, env):
        specializer = self.make_specializer(transforms.LateSpecializer, ast,
                                            env)
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
        fixer = self.make_specializer(FixMissingLocations, ast, env)
        fixer.visit(ast)
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

class LinkingStage(PipelineStage):
    """
    Link the resulting LLVM function into the global fat module.
    """

    def transform(self, ast, env):
        func_env = env.translation.crnt

        # Link libraries into module
        env.context.intrinsic_library.link(func_env.lfunc.module)
        # env.context.cbuilder_library.link(func_env.lfunc.module)

        if func_env.link:
            # Link function into fat LLVM module
            func_env.lfunc = env.llvm_context.link(func_env.lfunc)
            func_env.translator.lfunc = func_env.lfunc

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
                func_env.translator.build_wrapper_function(get_lfunc=True))
            func_env.numba_wrapper_func = numbawrapper
            func_env.llvm_wrapper_func = lfuncwrapper

        return ast

class ErrorReporting(PipelineStage):
    "Sort and issue warnings and errors"
    def transform(self, ast, env):
        error_env = env.translation.crnt.error_env
        post_mortem = error_env.enable_post_mortem
        reporting.report(error_env, post_mortem=post_mortem)
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
