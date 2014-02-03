import numba
from numba import error
from numba.exttypes import utils
from numba.exttypes import jitclass
from numba.exttypes import autojitclass
from numba.exttypes.autojitclass import autojit_class_wrapper

from llvm import core as _lc

#------------------------------------------------------------------------
# Build Extension Type (@jit)
#------------------------------------------------------------------------

def jit_extension_class(py_class, translator_kwargs, env):
    llvm_module = translator_kwargs.get('llvm_module', None)
    if llvm_module is None:
        llvm_module = _lc.Module.new('tmp.extension_class.%X' % id(py_class))
        translator_kwargs['llvm_module'] = llvm_module

    return jitclass.create_extension(env, py_class, translator_kwargs)

#------------------------------------------------------------------------
# Build Dynamic Extension Type (@autojit)
#------------------------------------------------------------------------

def autojit_extension_class(env, py_class, flags, argtypes):
    """
    Compile an extension class given the NumbaEnvironment and the Python
    class that contains the functions that are to be compiled.
    """
    return autojitclass.create_extension(env, py_class, flags, argtypes)
