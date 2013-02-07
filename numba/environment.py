import types

from numba.functions import FunctionCache
from numba.utils import TypedProperty, NumbaContext

# ______________________________________________________________________

class _AbstractNumbaEnvironment(object):
    '''Used to break circular type dependency.'''

# ______________________________________________________________________

class FunctionEnvironment(object):
    '''State for a function under translation.'''

# ______________________________________________________________________

class TranslationEnvironment(object):
    '''State for a given translation.'''
    # ____________________________________________________________
    # Properties

    numba = TypedProperty(_AbstractNumbaEnvironment, 'Parent environment')

    crnt = TypedProperty(
        FunctionEnvironment,
        'The current function being processed by the pipeline.')

    stack = TypedProperty(list, 'A stack consisting of FunctionEnvironment '
                          'instances.  Used to manage lexical closures.')

    functions = TypedProperty(dict, 'map from functions under compilation to '
                              'FunctionEnvironments')

# ______________________________________________________________________

class PyccEnvironment(object):
    '''pycc environment

    Includes flags, and modules for exported functions.
    '''
    # ____________________________________________________________
    # Properties

    wrap = TypedProperty(
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

    def __init__(self, *args, **kws):
        self.reset(*args, **kws)

    def reset(self, *args, **kws):
        '''Clear the current set of exported functions.'''
        self.wrap = False
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

    default_pipeline = TypedProperty(
        str,
        'Default entry point name.  Used to index into the "pipelines" map.',
        'numba')

    context = TypedProperty(
        NumbaContext,
        'Defines a global typing context for handling promotions and type '
        'representations.')

    specializations = TypedProperty(
        FunctionCache, 'Cache for previously specialized functions.')

    exports = TypedProperty(
        PyccEnvironment, 'Translation environment for pycc usage')

    debug = TypedProperty(bool, 'Global debugging flag.', False)

    verify_stages = TypedProperty(
        bool,
        'Global flag for enabling detailed checks in translation pipeline '
        'stages.',
        False)

    translation = TypedProperty(
        (TranslationEnvironment, types.NoneType),
        'Current translation environment, specific to the current pipeline '
        'being run.')

    # ____________________________________________________________
    # Class members

    environment_map = {}

    # ____________________________________________________________
    # Methods

    def __init__(self, *args, **kws):
        self.pipelines = {
            self.default_pipeline : [
                'resolve_templates',
                'validate_signature',
                'ControlFlowAnalysis',
                #'ConstFolding',
                'TypeInfer',
                'TypeSet',
                #'dump_cfg',
                'ClosureTypeInference',
                'TransformFor',
                'Specialize',
                'LateSpecializer',
                'FixASTLocations',
                'cleanup_symtab',
                'CodeGen',
                ]}
        self.context = NumbaContext()
        self.specializations = FunctionCache(self.context)
        self.exports = PyccEnvironment()

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

# ______________________________________________________________________
# Main (self-test) routine

def main(*args):
    env = NumbaEnvironment.get_environment()
    env.translation = TranslationEnvironment()
    env.translation.numba = env
    env.translation = None

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
