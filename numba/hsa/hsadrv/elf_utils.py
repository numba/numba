"""
Implement the elf utils used to load binary brig-elf files.
"""

from __future__ import absolute_import, divide, print_function

import ctypes
import os
from . import drvapi

def _load_elf_utils_syms():
    """
    find the .so containing the implementation and export the symbols
    we want.
    """
    path = os.path.dirname(__file__)
    lib_path = os.path.join(path, '_hsa_support.so')
    lib = ctypes.CDLL(lib_path)
    PTR = ctypes.POINTER

    create = lib.create_brig_module_from_brig_file
    create.restype = drvapi.status_t
    create.argtypes = [ctypes.c_char_p, PTR(PTR(drvapi.hsa_ext_brig_module_t))]
    destroy = lib.destroy_brig_module
    destroy.restype = None
    destroy.argtypes = [PTR(PTR(drvapi.hsa_ext_brig_module_t))]
    return create, destroy


# export our desired symbols
create_brig_module_from_brig_file, destroy_brig_module = _load_elf_utils_syms()

del _load_elf_utils_syms
