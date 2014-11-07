import sys
import ctypes
import struct as struct_
from llvmlite.llvmpy.core import Type, Constant

_trace_refs_ = hasattr(sys, 'getobjects')
_plat_bits = struct_.calcsize('@P') * 8

_int8 = Type.int(8)
_int32 = Type.int(32)

_void_star = Type.pointer(_int8)

_int8_star = _void_star

_sizeof_py_ssize_t = ctypes.sizeof(getattr(ctypes, 'c_size_t'))
_llvm_py_ssize_t = Type.int(_sizeof_py_ssize_t * 8)

if _trace_refs_:
    _pyobject_head = Type.struct([_void_star, _void_star,
                                  _llvm_py_ssize_t, _void_star])
    _pyobject_head_init = Constant.struct([
        Constant.null(_void_star),            # _ob_next
        Constant.null(_void_star),            # _ob_prev
        Constant.int(_llvm_py_ssize_t, 1),    # ob_refcnt
        Constant.null(_void_star),            # ob_type
        ])

else:
    _pyobject_head = Type.struct([_llvm_py_ssize_t, _void_star])
    _pyobject_head_init = Constant.struct([
        Constant.int(_llvm_py_ssize_t, 1),    # ob_refcnt
        Constant.null(_void_star),            # ob_type
        ])

_pyobject_head_p = Type.pointer(_pyobject_head)
