'''llvm_types

Utility module containing common (to Numba) LLVM types.
'''
# ______________________________________________________________________

import sys

import llvm.core as lc

from numba import _ext

# ______________________________________________________________________

if sys.maxint > 2**33:
    _plat_bits = 64
else:
    _plat_bits = 32

_int1 = lc.Type.int(1)
_int8 = lc.Type.int(8)
_int32 = lc.Type.int(32)
_int64 = lc.Type.int(64)
_llvm_py_ssize_t = lc.Type.int(_ext.sizeof_py_ssize_t() * 8)
_intp = lc.Type.int(_plat_bits)
_intp_star = lc.Type.pointer(_intp)
_void_star = lc.Type.pointer(lc.Type.int(8))
_void_star_star = lc.Type.pointer(_void_star)
_float = lc.Type.float()
_double = lc.Type.double()
_complex64 = lc.Type.struct([_float, _float])
_complex128 = lc.Type.struct([_double, _double])

_pyobject_head = [_intp, lc.Type.pointer(_int32)]
_pyobject_head_struct = lc.Type.struct(_pyobject_head)
_pyobject_head_struct_p = lc.Type.pointer(_pyobject_head_struct)

if hasattr(sys, 'getobjects'):
    _trace_refs_ = True
    _pyobject_head = [lc.Type.pointer(_int32),
                      lc.Type.pointer(_int32)] + \
                      _pyobject_head
else:
    _trace_refs_ = False

_head_len = len(_pyobject_head)
_numpy_struct = lc.Type.struct(_pyobject_head+\
      [_void_star,          # data
       _int32,              # nd
       _intp_star,          # dimensions
       _intp_star,          # strides
       _void_star,          # base
       _void_star,          # descr
       _int32,              # flags
       _void_star,          # weakreflist
       _void_star,          # maskna_dtype
       _void_star,          # maskna_data
       _intp_star,          # masna_strides
      ])
_numpy_array = lc.Type.pointer(_numpy_struct)

_BASE_ARRAY_FIELD_OFS = len(_pyobject_head)

_numpy_array_field_ofs = {
    'data' : _BASE_ARRAY_FIELD_OFS,
    'ndim' : _BASE_ARRAY_FIELD_OFS + 1,
    'shape' : _BASE_ARRAY_FIELD_OFS + 2,
    'strides' : _BASE_ARRAY_FIELD_OFS + 3,
    # Skipping base for now...
    'descr' : _BASE_ARRAY_FIELD_OFS + 5,
}

# ______________________________________________________________________
# End of llvm_types.py
