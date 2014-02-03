import sys
import ctypes
import struct as struct_
from llvm.core import Type

_trace_refs_ = hasattr(sys, 'getobjects')
_plat_bits = struct_.calcsize('@P') * 8

_int8 = Type.int(8)
_int32 = Type.int(32)

_void_star = Type.pointer(_int8)

_int8_star = _void_star
_pyobject_head_struct_p = _void_star

_sizeof_py_ssize_t = ctypes.sizeof(getattr(ctypes, 'c_size_t'))
_llvm_py_ssize_t = Type.int(_sizeof_py_ssize_t * 8)