from llvm import core as _lc
import numba
from numba import pipeline, error, symtab, typesystem
from numba import typesystem
from numba.exttypes import extension_types
from numba.exttypes.jitclass import create_extension

from numba.minivect import minitypes

from numba.exttypes import logger

#------------------------------------------------------------------------
# Build Extension Type (@jit)
#------------------------------------------------------------------------

def jit_extension_class(py_class, translator_kwargs, env):
    llvm_module = translator_kwargs.get('llvm_module', None)
    if llvm_module is None:
        llvm_module = _lc.Module.new('tmp.extension_class.%X' % id(py_class))
        translator_kwargs['llvm_module'] = llvm_module

    return create_extension(env, py_class, translator_kwargs)

#------------------------------------------------------------------------
# Build Dynamic Extension Type (@autojit)
#------------------------------------------------------------------------

def create_dynamic_extension(env, py_class, flags):
    """
    Compile an extension class given the NumbaEnvironment and the Python
    class that contains the functions that are to be compiled.
    """
    raise NotImplementedError
