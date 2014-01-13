# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import weakref
import ast as ast_module
import types
import logging
import pprint
import collections

import llvm.core

from numba import pipeline, naming, error, reporting, PY3
from numba.utils import TypedProperty, WriteOnceTypedProperty, NumbaContext
from numba import functions, symtab
from numba.typesystem import TypeSystem, numba_typesystem, function
from numba.utility.cbuilder import library
from numba.nodes import metadata
from numba.control_flow import ControlFlow
from numba.codegen import translate
from numba.codegen import globalconstants
from numba.ndarray_helpers import NumpyArray

from numba.intrinsic import default_intrinsic_library
from numba.external import default_external_library
from numba.external.utility import default_utility_library

# ______________________________________________________________________
# Module data

logger = logging.getLogger(__name__)

if PY3:
    NoneType = type(None)
    name_types = str
else:
    NoneType = types.NoneType
    name_types = (str, unicode)

default_normalize_order = [
    'ast3to2',
    'resolve_templates',
    'validate_signature',
    'update_signature',
    'create_lfunc1',
    'NormalizeASTStage',
    'TransformBuiltinLoops',
    'ValidateASTStage',
]

default_cf_pipeline_order = ['ast3to2', 'ControlFlowAnalysis']


default_pipeline_order = default_normalize_order + [
    'ExpandPrange',
    'RewritePrangePrivates',
    'FixASTLocations',
    'ControlFlowAnalysis',
    'dump_cfg',
    #'ConstFolding',
    # 'dump_ast',
    'UpdateAttributeStatements',
    'TypeInfer',
    'CleanupPrange',
    'update_signature',
    'create_lfunc2',
    'TypeSet',
    'ClosureTypeInference',
    'create_lfunc3',
    'TransformFor',
    'Specialize',
    'RewriteArrayExpressions',
    'SpecializeComparisons',
    'SpecializeSSA',
    'SpecializeClosures',
    'Optimize',
    'SpecializeLoops',
    'LowerRaise',
    'FixASTLocations',
    'LateSpecializer',
    'ExtensionTypeLowerer',
    'SpecializeFunccalls',
    'SpecializeExceptions',
    'cleanup_symtab',
    'validate_arrays',
    'dump_ast',
    'FixASTLocations',
    'CodeGen',
    'dump_annotations',
    'dump_llvm',
    'PostPass',
    'LinkingStage',
    'dump_optimized',
    'WrapperStage',
    'ErrorReporting',
]


default_dummy_type_infer_pipeline_order = [
    'ast3to2',
    'TypeInfer',
    'TypeSet',
]

default_numba_lower_pipeline_order = [
    'ast3to2',
    'LateSpecializer',
    'SpecializeFunccalls',
    'SpecializeExceptions',
]


default_numba_wrapper_pipeline_order = default_numba_lower_pipeline_order

default_numba_late_translate_pipeline_order = \
    default_numba_lower_pipeline_order + [
    'CodeGen',
]

upto = lambda order, x: order[:order.index(x)+1]
upfr = lambda order, x: order[order.index(x)+1:]

default_type_infer_pipeline_order = upto(default_pipeline_order, 'TypeInfer')
default_compile_pipeline_order = upfr(default_pipeline_order, 'TypeInfer')
default_codegen_pipeline = upto(default_pipeline_order, 'CodeGen')
default_post_codegen_pipeline = upfr(default_pipeline_order, 'CodeGen')

# ______________________________________________________________________
# Convenience functions

def insert_stage(pipeline_order, stage, after=None, before=None):
    if after is not None:
        idx = pipeline_order.index(after) + 1
    else:
        idx = pipeline_order.index(before)

    pipeline_order.insert(idx, stage)

# ______________________________________________________________________
# Class definitions

class _AbstractNumbaEnvironment(object):
    '''Used to break circular type dependency between the translation
    and function environments and the top-level NumbaEnvironment.'''

# ______________________________________________________________________

class FunctionErrorEnvironment(object):
    """
    Environment for errors or warnings that occurr during translation of
    a function.
    """

    func = WriteOnceTypedProperty(
        (NoneType, types.FunctionType),
        'Function (or similar) being translated.')

    ast = TypedProperty(
        ast_module.AST,
        'Original Abstract Syntax Tree for the function being translated.')

    source = TypedProperty(
        list, #(str, unicode),
        "Function source code")

    enable_post_mortem = TypedProperty(
        bool,
        "Enable post-mortem debugging for the Numba compiler",
        False,
    )

    collection = TypedProperty(
        reporting.MessageCollection,
        "Collection of error and warning messages")

    warning_styles = {
        'simple' : reporting.MessageCollection,
        'fancy': reporting.FancyMessageCollection,
    }

    def __init__(self, func, ast, warnstyle):
        self.func = func
        self.ast = ast # copy.deepcopy(ast)

        # Retrieve the source code now
        source_descr = reporting.SourceDescr(func, ast)
        self.source = source_descr.get_lines()

        collection_cls = self.warning_styles[warnstyle]
        self.collection = collection_cls(self.ast, self.source)

    def merge_in(self, parent_error_env):
        """
        Merge error messages into another error environment.
        Useful to propagate error messages for inner functions outwards.
        """
        parent_error_env.collection.messages.extend(self.collection.messages)
        del self.collection.messages[:]

# ______________________________________________________________________

class FunctionEnvironment(object):
    '''State for a function under translation.'''
    # ____________________________________________________________
    # Properties

    numba = WriteOnceTypedProperty(
        _AbstractNumbaEnvironment,
        'Grandparent environment (top-level Numba environment).')

    func = WriteOnceTypedProperty(
        object, 'Function (or similar) being translated.')

    ast = TypedProperty(
        ast_module.AST,
        'Abstract syntax tree for the function being translated.')

    func_signature = TypedProperty(
        function,
        'Type signature for the function being translated.')

    is_partial = TypedProperty(
        bool,
        "Whether this environment is a partially constructed environment",
        False)

    func_name = TypedProperty(str, 'Target function name.')
    module_name = TypedProperty(str, 'Name of the function module.')

    mangled_name = TypedProperty(str, 'Mangled name of compiled function.')

    qualified_name = TypedProperty(str, "Target qualified function name "
                                        "('mymodule.myfunc')")

    llvm_module = TypedProperty(
        llvm.core.Module,
        'LLVM module for this function.  This module is first optimized and '
        'then linked into a global module.  The Python wrapper function goes '
        'directly into the main fat module.')

    error_env = TypedProperty(
        FunctionErrorEnvironment,
        "Error environment for this function.")

    lfunc = TypedProperty(
        (NoneType, llvm.core.Function),
        "Compiled, native, Numba function",
        None)

    lfunc_pointer = TypedProperty(
        (int, long) if not PY3 else int,
        "Pointer to underlying compiled function. Can be used as a callback.",
    )

    link = TypedProperty(
        bool,
        'Flag indicating whether the LLVM function needs to be linked into '
        'the global fast module from LLVMContextManager',
        True)

    wrap = TypedProperty(
        bool,
        'Flag indicating whether the function needs a wrapper function to be '
        'callable from Python.',
        True)

    llvm_wrapper_func = TypedProperty(
        (llvm.core.Function, NoneType),
        'The LLVM wrapper function for the target function.  This is a '
        'wrapper function that accept python object arguments and returns an '
        'object.')

    numba_wrapper_func = TypedProperty(
        object,
        'The Numba wrapper function (see numbafunction.c) for the target '
        'function.  This is a wrapper function that accept python object '
        'arguments and returns an object.')

    symtab = TypedProperty(
        (symtab.Symtab, dict),
        'A map from local variable names to symbol table variables for all '
        'local variables. '
        '({ "local_var_name" : numba.symtab.Variable(local_var_type) })')

    function_globals = TypedProperty(
        (dict, NoneType),
        "Globals dict of the function",)

    locals = TypedProperty(
        dict,
        'A map from local variable names to types.  Used to handle the locals '
        'keyword argument to the autojit decorator. '
        '({ "local_var_name" : local_var_type } for @autojit(locals=...))')

    template_signature = TypedProperty(
        (function, NoneType),
        'Template signature for @autojit.  E.g. T(T[:, :]).  See '
        'numba.typesystem.templatetypes.')

    typesystem = TypedProperty(TypeSystem, "Typesystem for this compilation")
    array = TypedProperty(object, "Array abstraction", NumpyArray)

    ast_metadata = TypedProperty(
        object,
        'Metadata for AST nodes of the function being compiled.')

    warn = True
    flow = TypedProperty(
        (NoneType, ControlFlow),
        "Control flow graph. See numba.control_flow.",
        default=None)

    # FIXME: Get rid of this.  See comment for translator property,
    # below.
    cfg_transform = TypedProperty(
        object, # Should be ControlFlowAnalysis.
        'The Control Flow Analysis transform object '
        '(control_flow.ControlFlowAnalysis). Set during the cfg pass.')
    cfdirectives = TypedProperty(
        dict, "Directives for control flow.",
        default={
            'warn.maybe_uninitialized': warn,
            'warn.unused_result': False,
            'warn.unused': warn,
            'warn.unused_arg': warn,
            # Set the below flag to a path to generate CFG dot files
            'control_flow.dot_output': os.path.expanduser("~/cfg.dot"),
            'control_flow.dot_annotate_defs': False,
        },
    )

    kill_attribute_assignments = TypedProperty( # Prange
        (set, frozenset),
        "Assignments to attributes that need to be removed from type "
        "inference pre-analysis. We need to do this for prange since we "
        "need to infer the types of variables to build a struct type for "
        "those variables. So we need type inference to be properly ordered, "
        "and not look at the attributes first.")

    # FIXME: Get rid of this property; pipeline stages are users and
    # transformers of the environment.  Any state needed beyond a
    # given stage should be put in the environment instead of keeping
    # around the whole transformer.
    # TODO: Create linking stage
    translator = TypedProperty(
        object, # FIXME: Should be LLVMCodeGenerator, but that causes
                # module coupling.
        'The code generator instance used to generate the target LLVM '
        'function.  Set during the code generation pass, and used for '
        'after-the-fact wrapper generation.')

    is_closure = TypedProperty(
        bool,
        'Flag indicating if the current function under translation is a '
        'closure or not.',
        False)

    closures = TypedProperty(
        dict, 'Map from ast nodes to closures.')

    closure_scope = TypedProperty(
        (dict, NoneType),
        'Collective symtol table containing all entries from outer '
        'functions.')

    need_closure_wrapper = TypedProperty(
        bool, "Whether this closure needs a Python wrapper function",
        default=True,
    )

    refcount_args = TypedProperty(
        bool, "Whether to use refcounting for the function arguments", True)

    warn = TypedProperty(
        bool,
        'Flag that enables control flow warnings on a per-function level.',
        True)

    annotations = TypedProperty(
        dict, "Annotation dict { lineno : Annotation }"
    )

    intermediates = TypedProperty(
        list, "list of Intermediate objects for annotation",
    )

    warnstyle = TypedProperty(
        str if PY3 else basestring,
        'Warning style, currently available: simple, fancy',
        default='fancy'
    )

    postpasses = TypedProperty(
        dict,
        "List of passes that should run on the final llvm ir before linking",
    )

    kwargs = TypedProperty(
        dict,
        'Additional keyword arguments.  Deprecated, but kept for backward '
        'compatibility.')

    # ____________________________________________________________
    # Methods

    def __init__(self, *args, **kws):
        self.init(*args, **kws)

    def init(self, parent, func, ast, func_signature,
             name=None, qualified_name=None,
             mangled_name=None,
             llvm_module=None, wrap=True, link=True,
             symtab=None,
             error_env=None, function_globals=None, locals=None,
             template_signature=None, is_closure=False,
             closures=None, closure_scope=None,
             refcount_args=True,
             ast_metadata=None, warn=True, warnstyle='fancy',
             typesystem=None, array=None, postpasses=None, annotate=False,
             **kws):

        self.parent = parent
        self.numba = parent.numba
        self.func = func
        self.ast = ast
        self.func_signature = func_signature

        if name is None:
            if self.func:
                name = self.func.__name__
            else:
                name = self.ast.name

        if self.func and self.func.__module__:
            qname = '.'.join([self.func.__module__, name])
        else:
            qname = name

        if function_globals is not None:
            self.function_globals = function_globals
        else:
            self.function_globals = self.func.__globals__

        if self.func:
            self.module_name = self.func.__module__ or '<unamed.module>'
        else:
            self.module_name = self.function_globals.get("__name__", "")

        if mangled_name is None:
            mangled_name = naming.specialized_mangle(qname,
                                                     self.func_signature.args)

        self.func_name = name
        self.mangled_name = mangled_name
        self.qualified_name = qualified_name or name
        self.llvm_module = (llvm_module if llvm_module
                                 else self.numba.llvm_context.module)

        self._annotate = annotate
        self.wrap = wrap
        self.link = link
        self.llvm_wrapper_func = None
        self.symtab = symtab if symtab is not None else {}

        self.error_env = error_env or FunctionErrorEnvironment(self.func,
                                                               self.ast,
                                                               warnstyle)


        self.locals = locals if locals is not None else {}
        self.template_signature = template_signature
        self.is_closure = is_closure
        self.closures = closures if closures is not None else {}
        self.closure_scope = closure_scope
        self.kill_attribute_assignments = set()

        self.refcount_args = refcount_args
        self.typesystem = typesystem or numba_typesystem
        if array:
            self.array = array
            # assert issubclass(array, NumpyArray)

        import numba.postpasses
        self.postpasses = postpasses or numba.postpasses.default_postpasses

        if ast_metadata is not None:
            self.ast_metadata = ast_metadata
        else:
            self.ast_metadata = metadata.create_metadata_env()

        self.annotations = collections.defaultdict(list)
        self.intermediates = []
        self.warn = warn
        self.warnstyle = warnstyle
        self.kwargs = kws

    def getstate(self):
        state = dict(
            parent=self.parent,
            func=self.func,
            ast=self.ast,
            func_signature=self.func_signature,
            name=self.func_name,
            mangled_name=self.mangled_name,
            qualified_name=self.qualified_name,
            llvm_module=self.llvm_module,
            wrap=self.wrap,
            link=self.link,
            symtab=self.symtab,
            function_globals=self.function_globals,
            locals=self.locals,
            template_signature=self.template_signature,
            is_closure=self.is_closure,
            closures=self.closures,
            kill_attribute_assignments=self.kill_attribute_assignments,
            closure_scope=self.closure_scope,
            warn=self.warn,
            warnstyle=self.warnstyle,
            postpasses=self.postpasses,
        )
        return state

    def inherit(self, **kwds):
        """
        Inherit from a parent FunctionEnvironment (e.g. to run pipeline stages
        on a subset of the AST).
        """
        # TODO: link these things together
        state = self.getstate()
        state.update(kwds)
        return type(self)(**state)

    @property
    def annotate(self):
        "Whether we need to annotate the source"
        return self._annotate or self.numba.cmdopts.get('annotate')

    @property
    def func_doc(self):
        if self.func is not None:
            return self.func.__doc__
        else:
            return ast_module.get_docstring(self.ast)

# ______________________________________________________________________

class TranslationEnvironment(object):
    '''State for a given translation.'''
    # ____________________________________________________________
    # Properties

    numba = TypedProperty(_AbstractNumbaEnvironment, 'Parent environment')

    crnt = TypedProperty(
        (FunctionEnvironment, NoneType),
        'The environment corresponding to the current function under '
        'translation.')

    stack = TypedProperty(
        list,
        'A stack consisting of FunctionEnvironment instances.  Used to '
        'manage lexical closures.')

    functions = TypedProperty(
        dict,
        'A map from target function names that are under compilation to their '
        'corresponding FunctionEnvironments')

    func_envs = TypedProperty(
        weakref.WeakKeyDictionary,
        "Map from root AST nodes to FunctionEnvironment objects."
        "This allows tracking environments of partially processed ASTs.")

    nopython = TypedProperty(
        bool,
        'Flag used to indicate if calls to the Python C API are permitted or '
        'not during code generation.',
        False)

    allow_rebind_args = TypedProperty(
        bool,
        'Flag indicating whether the type of arguments may be overridden for '
        '@jit functions.  This is always true (except for in tests perhaps!)',
        True)

    warn = TypedProperty(
        bool,
        'Flag that enables control flow warnings. FunctionEnvironment inherits '
        'this unless overridden.',
        True)

    is_pycc = TypedProperty(
        bool,
        'Flag that tells us whether this function is being exported with pycc.',
        False)

    # ____________________________________________________________
    # Methods

    def __init__(self, parent, **kws):
        self.numba = parent
        self.crnt = None
        self.stack = [(kws, None)]
        self.functions = {}
        self.func_envs = weakref.WeakKeyDictionary()
        self.set_flags(**kws)

    def set_flags(self, **kws):
        self.nopython = kws.get('nopython', False)
        self.allow_rebind_args = kws.get('allow_rebind_args', True)
        self.warn = kws.get('warn', True)
        self.is_pycc = kws.get('is_pycc', False)

    def get_or_make_env(self, func, ast, func_signature, **kwds):
        if ast not in self.func_envs:
            kwds.setdefault('warn', self.warn)
            func_env = self.numba.FunctionEnvironment(
                    self, func, ast, func_signature, **kwds)
            self.func_envs[ast] = func_env
        else:
            func_env = self.func_envs[ast]
            if func_env.is_partial:
                state = func_env.partial_state
            else:
                state = func_env.getstate()

            state.update(kwds, func=func, ast=ast,
                               func_signature=func_signature)
            func_env.init(self, **state)

        return func_env

    def get_env(self, ast):
        if ast in self.func_envs:
            return self.func_envs[ast]
        else:
            return None

    def make_partial_env(self, ast, **kwds):
        """
        Create a partial environment for a function that only initializes
        the given attributes.

        Later attributes will override existing attributes.
        """
        if ast in self.func_envs:
            func_env = self.func_envs[ast]
        else:
            func_env = self.numba.FunctionEnvironment.__new__(
                    self.numba.FunctionEnvironment)
            func_env.is_partial = True
            func_env.partial_state = kwds

            for key, value in kwds.iteritems():
                setattr(func_env, key, value)

            self.func_envs[ast] = func_env
            func_env.ast = ast

        return func_env

    def push(self, func, ast, func_signature, **kws):
        func_env = self.get_or_make_env(func, ast, func_signature, **kws)
        return self.push_env(func_env, **kws)

    def push_env(self, func_env, **kws):
        self.set_flags(**kws)
        self.crnt = func_env
        self.stack.append((kws, self.crnt))
        self.functions[self.crnt.func_name] = self.crnt
        self.func_envs[func_env.ast] = func_env
        if self.numba.debug:
            logger.debug('stack=%s\ncrnt=%r (%r)', pprint.pformat(self.stack),
                         self.crnt, self.crnt.func if self.crnt else None)
        return self.crnt

    def pop(self):
        ret_val = self.stack.pop()
        kws, self.crnt = self.stack[-1]
        self.set_flags(**kws)
        if self.numba.debug:
            logger.debug('stack=%s\ncrnt=%r (%r)', pprint.pformat(self.stack),
                         self.crnt, self.crnt.func if self.crnt else None)
        return ret_val

# ______________________________________________________________________

class TranslationContext(object):
    """Context manager for handling a translation.  Pushes a
    FunctionEnvironment input onto the given translation environment's
    stack, and pops it when leaving the translation context.
    """
    def __init__(self, env, *args, **kws):
        self.translation_environment = env.translation
        self.args = args
        self.kws = kws

    def __enter__(self):
        return self.translation_environment.push(*self.args, **self.kws)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.translation_environment.pop()

# ______________________________________________________________________

class PyccEnvironment(object):
    '''pycc environment

    Includes flags, and modules for exported functions.
    '''
    # ____________________________________________________________
    # Properties

    wrap_exports = TypedProperty(
        bool,
        'Boolean flag used to indicate that Python wrappers should be '
        'generated for exported functions.')

    function_signature_map = TypedProperty(
        dict,
        'Map from function names to type signatures for the translated '
        'function (used for header generation).')

    function_module_map = TypedProperty(
        dict,
        'Map from function names to LLVM modules that define the translated '
        'function.')

    function_wrapper_map = TypedProperty(
        dict,
        'Map from function names to tuples containing LLVM wrapper functions '
        'and LLVM modules that define the wrapper function.')

    # ____________________________________________________________
    # Methods

    def __init__(self, wrap_exports=False, *args, **kws):
        self.reset(wrap_exports, *args, **kws)

    def reset(self, wrap_exports=False, *args, **kws):
        '''Clear the current set of exported functions.'''
        self.wrap_exports = wrap_exports
        self.function_signature_map = {}
        self.function_module_map = {}
        self.function_wrapper_map = {}

# ______________________________________________________________________

class NumbaEnvironment(_AbstractNumbaEnvironment):
    '''Defines global state for a Numba translator. '''
    # ____________________________________________________________
    # Properties

    name = TypedProperty(name_types, "Name of the environment.")

    pipelines = TypedProperty(
        dict, 'Map from entry point names to PipelineStages.')

    pipeline_stages = TypedProperty(
        types.ModuleType,
        'Namespace for pipeline stages.  Initially set to the numba.pipeline '
        'module.',
        pipeline)

    default_pipeline = TypedProperty(
        str,
        'Default entry point name.  Used to index into the "pipelines" map.',
        default='numba')

    context = TypedProperty(
        NumbaContext,
        'Defines a global typing context for handling promotions and type '
        'representations.')

    specializations = TypedProperty(
        functions.FunctionCache, 'Cache for previously specialized functions.')

    exports = TypedProperty(
        PyccEnvironment, 'Translation environment for pycc usage')

    debug = TypedProperty(
        bool,
        'Global flag indicating verbose debugging output should be enabled.',
        False)

    debug_coercions = TypedProperty(
        bool,
        'Flag for checking type coercions done during late specialization.',
        False)

    stage_checks = TypedProperty(
        bool,
        'Global flag for enabling detailed checks in translation pipeline '
        'stages.',
        False)

    translation = TypedProperty(
        TranslationEnvironment,
        'Current translation environment, specific to the current pipeline '
        'being run.')

    llvm_context = TypedProperty(
        translate.LLVMContextManager,
        "Manages the global LLVM module and linkages of new translations."
    )

    constants_manager = TypedProperty(
        globalconstants.LLVMConstantsManager,
        "Holds constant values in an LLVM module.",
        default=globalconstants.LLVMConstantsManager(),
    )

    cmdopts = TypedProperty(
        dict, "Dict of command line options from bin/numba.py", {},
    )

    annotation_blocks = TypedProperty(
        list, "List of annotation information for different functions."
    )

    # ____________________________________________________________
    # Class members

    environment_map = {}

    TranslationContext = TranslationContext
    TranslationEnvironment = TranslationEnvironment
    FunctionEnvironment = FunctionEnvironment

    # ____________________________________________________________
    # Methods

    def __init__(self, name, *args, **kws):
        self.name = name
        actual_default_pipeline = pipeline.ComposedPipelineStage(
            default_pipeline_order)
        self.pipelines = {
            self.default_pipeline : actual_default_pipeline,
            'normalize' : pipeline.ComposedPipelineStage(
                default_normalize_order),
            'cf' : pipeline.ComposedPipelineStage(
                default_cf_pipeline_order),
            'type_infer' : pipeline.ComposedPipelineStage(
                default_type_infer_pipeline_order),
            'dummy_type_infer' : pipeline.ComposedPipelineStage(
                default_dummy_type_infer_pipeline_order),
            'compile' : pipeline.ComposedPipelineStage(
                default_compile_pipeline_order),
            'wrap_func' : pipeline.ComposedPipelineStage(
                default_numba_wrapper_pipeline_order),
            'lower' : pipeline.ComposedPipelineStage(
                default_numba_lower_pipeline_order),
            'late_translate' : pipeline.ComposedPipelineStage(
                default_numba_late_translate_pipeline_order),
            'codegen' : pipeline.ComposedPipelineStage(
                default_codegen_pipeline),
            'post_codegen' : pipeline.ComposedPipelineStage(
                default_post_codegen_pipeline),
            }
        self.context = NumbaContext()
        self.specializations = functions.FunctionCache(env=self)
        self.exports = PyccEnvironment()
        self.translation = self.TranslationEnvironment(self)
        self.debug = logger.getEffectiveLevel() < logging.DEBUG

        # FIXME: NumbaContext has up to now been used as a stand in
        # for NumbaEnvironment, so the following member definitions
        # should be moved into the environment, and the code that uses
        # them should be updated.
        context = self.context
        context.env = self
        context.numba_pipeline = actual_default_pipeline
        context.function_cache = self.specializations
        context.intrinsic_library = default_intrinsic_library(context)
        context.external_library = default_external_library(context)
        context.utility_library = default_utility_library(context)
        self.llvm_context = translate.LLVMContextManager()
        self.annotation_blocks = []

    def link_cbuilder_utilities(self):
        self.context.cbuilder_library = library.CBuilderLibrary()
        self.context.cbuilder_library.declare_registered(self)

        # Link modules
        self.context.cbuilder_library.link(self.llvm_context.module)

    @classmethod
    def get_environment(cls, environment_key = None, *args, **kws):
        '''
        Given an optional key, return the global Numba environment for
        that key.  If no key is given, return the default global
        environment.

        Note that internally, the default environment is mapped to None.
        '''
        if environment_key in cls.environment_map:
            ret_val = cls.environment_map[environment_key]
        else:
            ret_val = cls(environment_key or 'numba', *args, **kws)
            cls.environment_map[environment_key] = ret_val
        return ret_val

    @property
    def crnt(self):
        return self.translation.crnt

    def get_pipeline(self, pipeline_name=None):
        '''Convenience function for getting a pipeline object (which
        should be a callable object that accepts (ast, env) arguments,
        and returns an ast).'''
        if pipeline_name is None:
            pipeline_name = self.default_pipeline
        return self.pipelines[pipeline_name]

    def get_or_add_pipeline(self, pipeline_name=None, pipeline_ctor=None):
        if pipeline_name is None:
            pipeline_name = self.default_pipeline
        if pipeline_name in self.pipelines:
            pipeline_obj = self.pipelines[pipeline_name]
        else:
            pipeline_obj = self.pipelines[pipeline_name] = pipeline_ctor()
        return pipeline_obj

    def __repr__(self):
        return "NumbaEnvironment(%s)" % self.name

# ______________________________________________________________________
# Main (self-test) routine

def main(*args):
    import numba as nb
    test_ast = ast_module.parse('def test_fn(a, b):\n  return a + b\n\n',
                                   '<string>', 'exec')
    exec(compile(test_ast, '<string>', 'exec'))
    test_fn_ast = test_ast.body[-1]
    test_fn_sig = nb.double(nb.double, nb.double)
    test_fn_sig.name = test_fn.__name__
    env = NumbaEnvironment.get_environment()
    with TranslationContext(env, test_fn, test_fn_ast, test_fn_sig):
        env.get_pipeline()(test_fn_ast, env)
    assert env.pipeline_stages == pipeline

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
