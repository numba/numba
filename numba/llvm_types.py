'''llvm_types

Utility module containing common (to Numba) LLVM types.
'''
# ______________________________________________________________________

import ctypes
import sys
import platform
import llvm.core as lc

import logging

logger = logging.getLogger(__name__)

# ______________________________________________________________________

_plat_bits = int(platform.architecture()[0][:2])

# Assuming sizeof(c_size_t) == sizeof(c_ssize_t) == sizeof(Py_ssize_t)...
_sizeof_py_ssize_t = ctypes.sizeof(
    getattr(ctypes, 'c_ssize_t', getattr(ctypes, 'c_size_t')))

_int1 = lc.Type.int(1)
_int8 = lc.Type.int(8)
_int8_star = lc.Type.pointer(_int8)
_int32 = lc.Type.int(32)
_int64 = lc.Type.int(64)
_llvm_py_ssize_t = lc.Type.int(_sizeof_py_ssize_t * 8)
_llvm_size_t = _llvm_py_ssize_t
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

class _LLVMCaster(object):
    def __init__(self, builder):
        self.builder = builder

    def cast(self, lvalue, dst_ltype):
        src_ltype = lvalue.type
        return self.build_cast(self.builder, lvalue, dst_ltype)

    def build_pointer_cast(_, builder, lval1, lty2):
        return builder.bitcast(lval1, lty2)

    def build_int_cast(_, builder, lval1, lty2, unsigned = False):
        width1 = lval1.type.width
        width2 = lty2.width
        ret_val = lval1
        if width2 > width1:
            if unsigned:
                ret_val = builder.zext(lval1, lty2)
            else:
                ret_val = builder.sext(lval1, lty2)
        elif width2 < width1:
            # JDR: Compromise here on logging level...
            logger.info("Warning: Perfoming downcast.  May lose information.")
            ret_val = builder.trunc(lval1, lty2)
        return ret_val

    def build_float_ext(_, builder, lval1, lty2):
        return builder.fpext(lval1, lty2)

    def build_float_trunc(_, builder, lval1, lty2):
        logger.info("Warning: Perfoming downcast.  May lose information.")
        return builder.fptrunc(lval1, lty2)

    def build_int_to_float_cast(_, builder, lval1, lty2, unsigned = False):
        ret_val = None
        if unsigned:
            ret_val = builder.uitofp(lval1, lty2)
        else:
            ret_val = builder.sitofp(lval1, lty2)
        return ret_val

    def build_float_to_int_cast(_, builder, lval1, lty2, unsigned = False):
        ret_val = None
        if unsigned:
            ret_val = builder.fptoui(lval1, lty2)
        else:
            ret_val = builder.fptosi(lval1, lty2)
        return ret_val

    CAST_MAP = {
        lc.TYPE_POINTER : build_pointer_cast,
        lc.TYPE_INTEGER: build_int_cast,

        (lc.TYPE_FLOAT, lc.TYPE_DOUBLE) : build_float_ext,
        (lc.TYPE_DOUBLE, lc.TYPE_FP128) : build_float_ext,

        (lc.TYPE_DOUBLE, lc.TYPE_FLOAT) : build_float_trunc,
        (lc.TYPE_FP128, lc.TYPE_DOUBLE) : build_float_trunc,

        (lc.TYPE_INTEGER, lc.TYPE_FLOAT) : build_int_to_float_cast,
        (lc.TYPE_INTEGER, lc.TYPE_DOUBLE) : build_int_to_float_cast,

        (lc.TYPE_FLOAT, lc.TYPE_INTEGER) : build_float_to_int_cast,
        (lc.TYPE_DOUBLE, lc.TYPE_INTEGER) : build_float_to_int_cast,

    }

    @classmethod
    def build_cast(cls, builder, lval1, lty2, *args, **kws):
        ret_val = lval1
        lty1 = lval1.type
        lkind1 = lty1.kind
        lkind2 = lty2.kind

        if lkind1 == lkind2:

            if lkind1 in cls.CAST_MAP:
                ret_val = cls.CAST_MAP[lkind1](cls, builder, lval1, lty2,
                                               *args, **kws)
            else:
                raise NotImplementedError(lkind1)
        else:
            map_index = (lkind1, lkind2)
            if map_index in cls.CAST_MAP:
                ret_val = cls.CAST_MAP[map_index](cls, builder, lval1, lty2,
                                                  *args, **kws)
            else:
                raise NotImplementedError(str(lty1), str(lty2))
        return ret_val

# ______________________________________________________________________
# End of llvm_types.py
