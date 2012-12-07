from numba import *
from .minivect import minitypes
from . import llvm_types

import logging
logger = logging.getLogger(__name__)

# TODO: Create a subclass of
# llpython.byte_translator.LLVMTranslator that does macro
# expansion.

def c_string_slice_2 (func_cache, builder, c_string, lb, ub = None):
    module = builder.basic_block.function.module
    logger.debug((func_cache, builder, c_string, lb, ub))
    _, CStringSlice2Len = func_cache.external_function_by_name('CStringSlice2Len',
                                                      module=module)
    _, CStringSlice2 = func_cache.external_function_by_name('CStringSlice2',
                                                   module=module)
    _, strlen = func_cache.external_function_by_name('strlen',
                                            module=module)
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

def c_string_slice_1 (func_cache, builder, c_string, lb):
    return c_string_slice_2(func_cache, builder, c_string, lb)

c_string_slice_1.__signature__ = minitypes.FunctionType(
    return_type = c_string_type,
    args = (c_string_type, Py_ssize_t))
