"""
Serialization support for compiled functions.
"""


from importlib.util import MAGIC_NUMBER as bc_magic

import marshal
import sys
from types import FunctionType, ModuleType

from numba.core import bytecode, compiler


#
# Pickle support
#

def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """
    return cls._rebuild(*args)


class _ModuleRef(object):

    def __init__(self, name):
        self.name = name

    def __reduce__(self):
        return _rebuild_module, (self.name,)


def _rebuild_module(name):
    if name is None:
        raise ImportError("cannot import None")
    __import__(name)
    return sys.modules[name]


def _get_function_globals_for_reduction(func):
    """
    Analyse *func* and return a dictionary of global values suitable for
    reduction.
    """
    func_id = bytecode.FunctionIdentity.from_function(func)
    bc = bytecode.ByteCode(func_id)
    globs = bc.get_used_globals()
    for k, v in globs.items():
        # Make modules picklable by name
        if isinstance(v, ModuleType):
            globs[k] = _ModuleRef(v.__name__)
    # Remember the module name so that the function gets a proper __module__
    # when rebuilding.  This is used to recreate the environment.
    globs['__name__'] = func.__module__
    return globs

def _reduce_function(func, globs):
    """
    Reduce a Python function and its globals to picklable components.
    If there are cell variables (i.e. references to a closure), their
    values will be frozen.
    """
    if func.__closure__:
        cells = [cell.cell_contents for cell in func.__closure__]
    else:
        cells = None
    return _reduce_code(func.__code__), globs, func.__name__, cells

def _reduce_code(code):
    """
    Reduce a code object to picklable components.
    """
    return marshal.version, bc_magic, marshal.dumps(code)

def _dummy_closure(x):
    """
    A dummy function allowing us to build cell objects.
    """
    return lambda: x

def _rebuild_function(code_reduced, globals, name, cell_values):
    """
    Rebuild a function from its _reduce_function() results.
    """
    if cell_values:
        cells = tuple(_dummy_closure(v).__closure__[0] for v in cell_values)
    else:
        cells = ()
    code = _rebuild_code(*code_reduced)
    modname = globals['__name__']
    try:
        _rebuild_module(modname)
    except ImportError:
        # If the module can't be found, avoid passing it (it would produce
        # errors when lowering).
        del globals['__name__']
    return FunctionType(code, globals, name, (), cells)

def _rebuild_code(marshal_version, bytecode_magic, marshalled):
    """
    Rebuild a code object from its _reduce_code() results.
    """
    if marshal.version != marshal_version:
        raise RuntimeError("incompatible marshal version: "
                           "interpreter has %r, marshalled code has %r"
                           % (marshal.version, marshal_version))
    if bc_magic != bytecode_magic:
        raise RuntimeError("incompatible bytecode version")
    return marshal.loads(marshalled)

