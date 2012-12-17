from numba import *
from intrinsic import Intrinsic
from llpython.byte_translator import LLVMTranslator

__all__ = ['CStringSlice2',
           'CStringSlice2Len']

class CStringSlice2 (Intrinsic):
    arg_types = [c_string_type, c_string_type, size_t, Py_ssize_t, Py_ssize_t]
    return_type = void

    def implementation(self, module, lfunc):
        # logger.debug((module, str(lfunc)))
        def _py_c_string_slice(out_string, in_string, in_str_len, lower,
                               upper):
            zero = lc_size_t(0)
            if lower < zero:
                lower += in_str_len
            if upper < zero:
                upper += in_str_len
            elif upper > in_str_len:
                upper = in_str_len
            temp_len = upper - lower
            if temp_len < zero:
                temp_len = zero
            strncpy(out_string, in_string + lower, temp_len)
            out_string[temp_len] = li8(0)
            return
        LLVMTranslator(module).translate(_py_c_string_slice,
                                         llvm_function = lfunc)
        return lfunc

class CStringSlice2Len(Intrinsic):
    arg_types = [c_string_type, size_t, Py_ssize_t, Py_ssize_t]
    return_type = size_t

    def implementation(self, module, lfunc):
        def _py_c_string_slice_len(in_string, in_str_len, lower, upper):
            zero = lc_size_t(0)
            if lower < zero:
                lower += in_str_len
            if upper < zero:
                upper += in_str_len
            elif upper > in_str_len:
                upper = in_str_len
            temp_len = upper - lower
            if temp_len < zero:
                temp_len = zero
            return temp_len + lc_size_t(1)
        LLVMTranslator(module).translate(_py_c_string_slice_len,
                                         llvm_function = lfunc)
        return lfunc
