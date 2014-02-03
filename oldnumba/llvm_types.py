# -*- coding: utf-8 -*-
'''llvm_types

Utility module containing common (to Numba) LLVM types.
'''
from __future__ import print_function, division, absolute_import
# ______________________________________________________________________

import ctypes
import struct as struct_
import llvm.core as lc

from numba import utils
from numba.typedefs import _trace_refs_, PyObject_HEAD
from numba.typesystem import numba_typesystem

import logging

logger = logging.getLogger(__name__)

# ______________________________________________________________________

_plat_bits = struct_.calcsize('@P') * 8

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


def to_llvm(type):
    return numba_typesystem.convert("llvm", type)
    # return type.to_llvm(utils.context)

_pyobject_head = [to_llvm(ty) for name, ty in PyObject_HEAD.fields]
_pyobject_head_struct = to_llvm(PyObject_HEAD)
_pyobject_head_struct_p = lc.Type.pointer(_pyobject_head_struct)

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
      ])
_numpy_array = lc.Type.pointer(_numpy_struct)

_BASE_ARRAY_FIELD_OFS = len(_pyobject_head)

_numpy_array_field_ofs = {
    'data' : _BASE_ARRAY_FIELD_OFS,
    'ndim' : _BASE_ARRAY_FIELD_OFS + 1,
    'shape' : _BASE_ARRAY_FIELD_OFS + 2,
    'strides' : _BASE_ARRAY_FIELD_OFS + 3,
    'base' : _BASE_ARRAY_FIELD_OFS + 4,
    'descr' : _BASE_ARRAY_FIELD_OFS + 5,
}

def constant_int(value, type=_int32):
    return lc.Constant.int(type, value)

# ______________________________________________________________________

class _LLVMCaster(object):
    def __init__(self, builder):
        self.builder = builder

    def cast(self, lvalue, dst_ltype, *args, **kws):
        src_ltype = lvalue.type
        if src_ltype == dst_ltype:
            return lvalue
        return self.build_cast(self.builder, lvalue, dst_ltype, *args, **kws)

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

        (lc.TYPE_FLOAT, lc.TYPE_DOUBLE)     : build_float_ext,
        (lc.TYPE_DOUBLE, lc.TYPE_FP128)     : build_float_ext,
        (lc.TYPE_DOUBLE, lc.TYPE_PPC_FP128) : build_float_ext,
        (lc.TYPE_DOUBLE, lc.TYPE_X86_FP80)  : build_float_ext,

        (lc.TYPE_DOUBLE, lc.TYPE_FLOAT)     : build_float_trunc,
        (lc.TYPE_FP128, lc.TYPE_DOUBLE)     : build_float_trunc,
        (lc.TYPE_PPC_FP128, lc.TYPE_DOUBLE) : build_float_trunc,
        (lc.TYPE_X86_FP80, lc.TYPE_DOUBLE)  : build_float_trunc,

        (lc.TYPE_INTEGER, lc.TYPE_FLOAT)    : build_int_to_float_cast,
        (lc.TYPE_INTEGER, lc.TYPE_DOUBLE)   : build_int_to_float_cast,
        (lc.TYPE_INTEGER, lc.TYPE_FP128)    : build_int_to_float_cast,
        (lc.TYPE_INTEGER, lc.TYPE_PPC_FP128): build_int_to_float_cast,
        (lc.TYPE_INTEGER, lc.TYPE_X86_FP80) : build_int_to_float_cast,

        (lc.TYPE_FLOAT, lc.TYPE_INTEGER)    : build_float_to_int_cast,
        (lc.TYPE_DOUBLE, lc.TYPE_INTEGER)   : build_float_to_int_cast,
        (lc.TYPE_FP128, lc.TYPE_INTEGER)    : build_float_to_int_cast,
        (lc.TYPE_PPC_FP128, lc.TYPE_INTEGER): build_float_to_int_cast,
        (lc.TYPE_X86_FP80, lc.TYPE_INTEGER) : build_float_to_int_cast,

        }

    @classmethod
    def build_cast(cls, builder, lval1, lty2, *args, **kws):
        ret_val = lval1
        lty1 = lval1.type
        lkind1 = lty1.kind
        lkind2 = lty2.kind

        # This looks like the wrong place to enforce this
        # TODO: We need to pass in the numba types instead
        # if lc.TYPE_INTEGER in (lkind1, lkind2) and 'unsigned' not in kws:
        #     # Be strict about having `unsigned` define when
        #     # we have integer types
        #     raise ValueError("Unknown signedness for integer type",
        #                      '%s -> %s' % (lty1, lty2), args, kws)

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
                raise NotImplementedError('Unable to cast from %s to %s.' %
                                          (str(lty1), str(lty2)))
        return ret_val

# ______________________________________________________________________
# End of llvm_types.py
