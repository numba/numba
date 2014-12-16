"""
Implement the elf utils used to load binary brig-elf files.
"""

from __future__ import absolute_import, print_function

import sys
import ctypes
import os
from . import drvapi

def _load_elf_utils_syms():
    """
    find the .so containing the implementation and export the symbols
    we want.
    """
    path = os.path.dirname(__file__)
    if sys.version_info[0] >= 3:
        lib_name = '_hsa_support.cpython-{0}{1}m.so'.format(
            *sys.version_info[:2])
    else:
        lib_name = '_hsa_support.so'
    lib_path = os.path.join(path, lib_name)
    lib = ctypes.CDLL(lib_path)
    PTR = ctypes.POINTER

    create = lib.create_brig_module_from_brig_file
    create.restype = drvapi.hsa_status_t
    create.argtypes = [ctypes.c_char_p, PTR(PTR(drvapi.hsa_ext_brig_module_t))]
    create.errcheck = drvapi._check_error

    destroy = lib.destroy_brig_module
    destroy.restype = None
    destroy.argtypes = [PTR(drvapi.hsa_ext_brig_module_t)]

    find_symbol_offset = lib.find_symbol_offset
    find_symbol_offset.restype = drvapi.hsa_status_t
    find_symbol_offset.argtypes = [PTR(drvapi.hsa_ext_brig_module_t),
                                   ctypes.c_char_p,
                                   PTR(drvapi.hsa_ext_brig_code_section_offset32_t)]
    find_symbol_offset.errcheck = drvapi._check_error

    return create, destroy, find_symbol_offset


# export our desired symbols
create_brig_module_from_brig_file, destroy_brig_module, find_symbol_offset = _load_elf_utils_syms()

del _load_elf_utils_syms
