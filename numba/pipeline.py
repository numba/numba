"""
This module contains the Pipeline class which provides a pluggable way to
define the transformations and the order in which they run on the AST.
"""

import inspect
import ast as ast_module
import logging
import functools
import pprint
import random
from timeit import default_timer as _timer

import llvm.core as lc

import numba.closure
from numba import error
from numba import functions, naming, transforms, control_flow
from numba import ast_type_inference as type_inference
from numba import ast_constant_folding as constant_folding
from numba import ast_translate
from numba import utils
from numba.utils import dump
from numba.asdl import schema
from numba.minivect import minitypes
import numba.visitors

logger = logging.getLogger(__name__)

def get_locals(ast, locals_dict):
    if locals_dict is None:
        locals_dict = getattr(ast, "locals_dict", {})
    elif hasattr(ast, "locals_dict"):
        assert ast.locals_dict is locals_dict

    ast.locals_dict = locals_dict
    return locals_dict

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
        'late_specializer',
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
        self.llvm_module = lc.Module.new(self.module_name(func))

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

    def module_name(self, func):
        if func is None:
            name = "NoneFunc"
            func_id = random.randrange(1000000)
        else:
            name = '%s.%s' % (func.__module__, func.__name__)
            func_id = id(func)

        return 'tmp.module.%s.%x' % (name, func_id)

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
                logger.info("%X pipeline stage %30s:\t%.3fms",
                            id(self), method_name, (te - ts) * 1000)

        tomega = _timer() # for profiling complete pipeline
        logger.info("%X pipeline entire:\t\t\t\t\t%.3fms",
                    id(self), (tomega - talpha) * 1000)

        return self.func_signature, self.symtab, ast

    #
    ### Pipeline stages
    #

    def resolve_templates(self, ast):
        # TODO: Unify with decorators module
        if self.template_signature is not None:
            from numba import typesystem

            argnames = [arg.id for arg in ast.args.args]
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
                            numba.closure.ClosureTypeInferer, ast,
                            warn=self.kwargs.get("warn", True))
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

    def cleanup_symtab(self, ast):
        "Pop original variables from the symtab"
        for var in ast.symtab.values():
            if not var.parent_var and var.renameable:
                ast.symtab.pop(var.name, None)

        return ast

    def codegen(self, ast):
        self.translator = self.make_specializer(ast_translate.LLVMCodeGenerator,
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
                                                argtypes, order=['cfg', 'type_infer'],
                                                **kwargs)
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

def compile_from_sig(context, func, signature, **kwds):
    return compile(context, func, signature.return_type, signature.args,
                   **kwds)

class PipelineStage(object):
    def check_preconditions(self, ast, env):
        return True

    def check_postconditions(self, ast, env):
        return True

    def transform(self, ast, env):
        raise NotImplementedError('%r does not implement transform!' %
                                  type(self))

    def make_specializer(self, cls, ast, env, **kws):
        return cls(env.context, env.crnt.func, ast,
                   func_signature=env.crnt.func_signature,
                   nopython=env.crnt.nopython,
                   symtab=env.crnt.symtab,
                   func_name=env.crnt.func_name,
                   llvm_module=env.crnt.llvm_module,
                   locals=env.crnt.locals,
                   allow_rebind_args=env.crnt.allow_rebind_args,
                   warn=env.crnt.warn,
                   **kws)

    def __call__(self, ast, env):
        if env.stage_checks: self.check_preconditions(ast, env)
        ast = self.transform(ast, env)
        if env.stage_checks: self.check_postconditions(ast, env)
        return ast

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
        env.crnt.symtab = transform.symtab
        ast.flow = transform.flow
        env.crnt.ast.cfg_transform = transform
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
        env.crnt.constvars = constvars
        const_folder = self.make_specializer(constant_folding.ConstantFolder,
                                             ast, env, constvars=constvars)
        return const_folder.visit(ast)

class TypeInfer(PipelineStage):
    def check_preconditions(self, ast, env):
        assert env.crnt.symtab is not None
        return super(TypeInfer, self).check_preconditions(ast, env)

    def transform(self, ast, env):
        type_inferer = self.make_specializer(type_inference.TypeInferer,
                                             ast, env,
                                             **env.crnt.kwargs)
        type_inferer.infer_types()
        env.crnt.func_signature = type_inferer.func_signature
        logger.debug("signature for %s: %s", env.crnt.func_name,
                     env.crnt.func_signature)
        env.crnt.symtab = type_inferer.symtab
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
                            numba.closure.ClosureTypeInferer, ast, env)
        return type_inferer.visit(ast)

class TransformFor(PipelineStage):
    def transform(self, ast, env):
        transform = self.make_specializer(transforms.TransformForIterable, ast,
                                          env)
        return transform.visit(ast)

class Specialize(PipelineStage):
    def transform(self, ast, env):
        return ast

class LateSpecializer(PipelineStage):
    def transform(self, ast, env):
        specializer = self.make_specializer(transforms.LateSpecializer, ast,
                                            env)
        return specializer.visit(ast)

class FixASTLocations(PipelineStage):
    def transform(self, ast, env):
        fixer = self.make_specializer(FixMissingLocations, ast, env)
        fixer.visit(ast)
        return ast

class CodeGen(PipelineStage):
    def transform(self, ast, env):
        env.crnt.translator = self.make_specializer(
            ast_translate.LLVMCodeGenerator, ast, env, **env.crnt.kwargs)
        env.crnt.translator.translate()
        return ast

class PipelineEnvironment(object):
    init_stages=[
        ControlFlowAnalysis,
        #ConstFolding,
        TypeInfer,
        TypeSet,
        ClosureTypeInference,
        TransformFor,
        Specialize,
        LateSpecializer,
        FixASTLocations,
        CodeGen,
        ]

    def __init__(self, parent=None, doc='', *args, **kws):
        self.reset(parent, doc, *args, **kws)

    @classmethod
    def init_env(cls, context, doc='', **kws):
        ret_val = cls(doc=doc)
        ret_val.context = context
        for stage in cls.init_stages:
            setattr(ret_val, stage.__name__, stage)
        #pipe = cls.init_stages[:]
        #pipe.reverse()
        #ret_val.pipeline = reduce(compose_stages, pipe)
        ret_val.pipeline = ComposedPipelineStage(cls.init_stages)
        ret_val.stage_checks = kws.pop('stage_checks', True)
        ret_val.__dict__.update(kws)
        ret_val.crnt = cls(ret_val)
        return ret_val

    def reset(self, parent=None, doc='', *args, **kws):
        self.__dict__ = {}
        super(PipelineEnvironment, self).__init__()
        self.parent = parent
        self.__doc__ = doc

    def init_func(self, func, ast, func_signature, **kws):
        self.reset(self.parent, self.__doc__)
        self.func = func
        self.ast = ast
        self.func_signature = func_signature
        self.func_name = kws.get('name')
        if not self.func_name:
            if func:
                module_name = inspect.getmodule(func).__name__
                name = '.'.join([module_name, func.__name__])
            else:
                name = ast.name
            self.func_name = naming.specialized_mangle(
                name, self.func_signature.args)
        self.symtab = kws.pop('symtab', {})
        self.llvm_module = kws.pop('llvm_module',
                                   self.parent.context.llvm_module)
        self.nopython = kws.pop('nopython', False)
        self.locals = kws.pop('locals', {})
        self.allow_rebind_args = kws.pop('allow_rebind_args', True)
        self.warn = kws.pop('warn', True)
        self.kwargs = kws

def check_stage(stage):
    if isinstance(stage, str):
        def _stage(ast, env):
            return getattr(env, stage)(ast, env)
        name = stage
        _stage.__name__ = stage
        stage = _stage
    elif isinstance(stage, type) and issubclass(stage, PipelineStage):
        name = stage.__name__
        stage = stage()
    else:
        name = stage.__name__
    return name, stage

def compose_stages(f0, f1):
    f0_name, f0 = check_stage(f0)
    f1_name, f1 = check_stage(f1)
    def _numba_pipeline_composition(ast, env):
        f1_result = f1(ast, env)
        f0_result = f0(f1_result, env)
        return f0_result
    name = '_o_'.join((f0_name, f1_name))
    _numba_pipeline_composition.__name__ = name
    return _numba_pipeline_composition

class ComposedPipelineStage(PipelineStage):
    def __init__(self, stages=None):
        if stages is None:
            stages = []
        self.stages = [check_stage(stage)[1] for stage in stages]

    def transform(self, ast, env):
        for stage in self.stages:
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
