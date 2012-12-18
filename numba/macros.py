from numba import *
from .minivect import minitypes
from . import llvm_types

import logging
logger = logging.getLogger(__name__)

# TODO: Create a subclass of
# llpython.byte_translator.LLVMTranslator that does macro
# expansion.

def c_string_slice_2 (context, builder, c_string, lb, ub = None):
    module = builder.basic_block.function.module
    logger.debug((context, builder, c_string, lb, ub))
    _, CStringSlice2Len = context.intrinsic_library.declare(module,
                                                           'CStringSlice2Len')
    _, CStringSlice2 = context.intrinsic_library.declare(module,
                                                        'CStringSlice2')
    _, strlen = context.external_library.declare(module, 'strlen')
    c_str_len = builder.call(strlen, [c_string])
    if ub is None:
        ub = c_str_len
    out_len = builder.call(CStringSlice2Len, [c_string, c_str_len, lb, ub])
    ret_val = builder.alloca_array(llvm_types._int8, out_len)
    builder.call(CStringSlice2, [ret_val, c_string, c_str_len, lb, ub])
    return ret_val

c_string_slice_2.__signature__ = minitypes.FunctionType(
    return_type = c_string_type,
    args = (c_string_type, Py_ssize_t, Py_ssize_t))

def c_string_slice_1 (context, builder, c_string, lb):
    return c_string_slice_2(context, builder, c_string, lb)

c_string_slice_1.__signature__ = minitypes.FunctionType(
    return_type = c_string_type,
    args = (c_string_type, Py_ssize_t))
