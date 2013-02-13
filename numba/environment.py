import ast as ast_module
import inspect
import types
import logging

import llvm.core

from numba import pipeline, naming, error
from numba.utils import TypedProperty, WriteOnceTypedProperty, NumbaContext
from numba.minivect.minitypes import FunctionType
from numba import functions, symtab

from numba.intrinsic import default_intrinsic_library
from numba.external import default_external_library
from numba.external.utility import default_utility_library

# ______________________________________________________________________
# Module data

logger = logging.getLogger(__name__)

default_pipeline_order = [
    'resolve_templates',
    'validate_signature',
    'ControlFlowAnalysis',
    #'ConstFolding',
    'TypeInfer',
    'TypeSet',
    'dump_cfg',
    'ClosureTypeInference',
    'TransformFor',
    'Specialize',
    'LateSpecializer',
    'FixASTLocations',
    'cleanup_symtab',
    'CodeGen',
]

default_type_infer_pipeline_order = [
    'ControlFlowAnalysis',
    'TypeInfer',
]

default_dummy_type_infer_pipeline_order = [
    'TypeInfer',
    'TypeSet',
]

# ______________________________________________________________________
# Class definitions

class _AbstractNumbaEnvironment(object):
    '''Used to break circular type dependency between the translation
    and function environments and the top-level NumbaEnvironment.'''

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
        FunctionType,
        'Type signature for the function being translated.')

    func_name = TypedProperty(str, 'Target function name.')

    llvm_module = TypedProperty(
        llvm.core.Module,
        'LLVM module for this function.  This module is first optimized and '
        'then linked into a global module.  The Python wrapper function goes '
        'directly into the main fat module.')

    wrap = TypedProperty(
        bool,
        'Flag indicating whether the function needs a wrapper function to be '
        'callable from Python.',
        True)

    llvm_wrapper_func = TypedProperty(
        (llvm.core.Function, types.NoneType),
        'The LLVM wrapper function for the target function.  This is a '
        'wrapper function that accept python object arguments and returns an '
        'object.')

    symtab = TypedProperty(
        (symtab.Symtab, dict),
        'A map from local variable names to symbol table variables for all '
        'local variables. '
        '({ "local_var_name" : numba.symtab.Variable(local_var_type) })')

    locals = TypedProperty(
        dict,
        'A map from local variable names to types.  Used to handle the locals '
        'keyword argument to the autojit decorator. '
        '({ "local_var_name" : local_var_type } for @autojit(locals=...))')

    template_signature = TypedProperty(
        object, # FIXME
        'Template signature for @autojit.  E.g. T(T[:, :]).  See '
        'numba.typesystem.templatetypes.')

    # FIXME: Get rid of this.  See comment for translator property,
    # below.
    cfg_transform = TypedProperty(
        object, # Should be ControlFlowAnalysis.
        'The Control Flow Analysis transform object '
        '(control_flow.ControlFlowAnalysis). Set during the cfg pass.')

    # FIXME: Get rid of this property; pipeline stages are users and
    # transformers of the environment.  Any state needed beyond a
    # given stage should be put in the environment instead of keeping
    # around the whole transformer.
    translator = TypedProperty(
        object, # FIXME: Should be LLVMCodeGenerator, but that causes
                # module coupling.
        'The code generator instance used to generate the target LLVM '
        'function.  Set during the code generation pass, and used for '
        'after-the-fact wrapper generation.')

    kwargs = TypedProperty(
        dict,
        'Additional keyword arguments.  Deprecated, but kept for backward '
        'compatibility.')

    # ____________________________________________________________
    # Methods

    def __init__(self, parent, func, ast, func_signature,
                 name=None, llvm_module=None, wrap=True, symtab=None,
                 locals=None, template_signature=None, cfg_transform=None,
                 **kws):
        self.numba = parent.numba
        self.func = func
        self.ast = ast
        self.func_signature = func_signature
        if name is None:
            if func:
                module_name = inspect.getmodule(func).__name__
                name = '.'.join([module_name, func.__name__])
            else:
                name = ast.name
            name = naming.specialized_mangle(name, func_signature.args)
        self.func_name = name
        self.llvm_module = (llvm_module if llvm_module
                            else self.numba.context.llvm_module)
        self.wrap = wrap
        self.llvm_wrapper_func = None
        self.symtab = symtab if symtab is not None else {}
        self.locals = locals if locals is not None else {}
        self.template_signature = template_signature
        self.cfg_transform = cfg_transform
        self.kwargs = kws

# ______________________________________________________________________

class TranslationEnvironment(object):
    '''State for a given translation.'''
    # ____________________________________________________________
    # Properties

    numba = TypedProperty(_AbstractNumbaEnvironment, 'Parent environment')

    crnt = TypedProperty(
        (FunctionEnvironment, types.NoneType),
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
        'Flag that enables control flow warnings.',
        True)

    # ____________________________________________________________
    # Methods

    def __init__(self, parent, **kws):
        self.numba = parent
        self.crnt = None
        self.stack = [(kws, None)]
        self.functions = {}
        self.set_flags(**kws)

    def set_flags(self, **kws):
        self.nopython = kws.get('nopython', False)
        self.allow_rebind_args = kws.get('allow_rebind_args', True)
        self.warn = kws.get('warn', True)

    def push(self, func, ast, func_signature, **kws):
        self.set_flags(**kws)
        self.crnt = FunctionEnvironment(self, func, ast, func_signature, **kws)
        self.stack.append((kws, self.crnt))
        self.functions[self.crnt.func_name] = self.crnt
        if self.numba.debug:
            logger.debug('stack=%r\ncrnt=%r (%r)', self.stack, self.crnt,
                         self.crnt.func if self.crnt else None)
        return self.crnt

    def pop(self):
        ret_val = self.stack.pop()
        kws, self.crnt = self.stack[-1]
        self.set_flags(**kws)
        if self.numba.debug:
            logger.debug('stack=%r\ncrnt=%r (%r)', self.stack, self.crnt,
                         self.crnt.func if self.crnt else None)
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
        'numba')

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

    # ____________________________________________________________
    # Class members

    environment_map = {}

    TranslationContext = TranslationContext

    # ____________________________________________________________
    # Methods

    def __init__(self, *args, **kws):
        actual_default_pipeline = pipeline.ComposedPipelineStage(
            default_pipeline_order)
        self.pipelines = {
            self.default_pipeline : actual_default_pipeline,
            'type_infer' : pipeline.ComposedPipelineStage(
                default_type_infer_pipeline_order),
            'dummy_type_infer' : pipeline.ComposedPipelineStage(
                default_dummy_type_infer_pipeline_order),
            }
        self.context = NumbaContext()
        self.specializations = functions.FunctionCache(self.context)
        self.exports = PyccEnvironment()
        self.translation = TranslationEnvironment(self)

        # FIXME: NumbaContext has up to now been used as a stand in
        # for NumbaEnvironment, so the following member definitions
        # should be moved into the environment, and the code that uses
        # them should be updated.
        context = self.context
        context.numba_pipeline = actual_default_pipeline
        context.function_cache = self.specializations
        context.intrinsic_library = default_intrinsic_library(context)
        context.external_library = default_external_library(context)
        context.utility_library = default_utility_library(context)

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
            ret_val = cls(*args, **kws)
            cls.environment_map[environment_key] = ret_val
        return ret_val

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

# ______________________________________________________________________
# Main (self-test) routine

def main(*args):
    import numba as nb
    test_ast = ast_module.parse('def test_fn(a, b):\n  return a + b\n\n',
                                   '<string>', 'exec')
    exec compile(test_ast, '<string>', 'exec')
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
