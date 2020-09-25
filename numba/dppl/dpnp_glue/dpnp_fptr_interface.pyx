# distutils: language = c++
# cython: language_level=3

import ctypes


cdef extern from "backend_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_NONE
        DPNP_FN_ADD
        DPNP_FN_ARGMAX
        DPNP_FN_ARGMIN
        DPNP_FN_ARGSORT
        DPNP_FN_CEIL
        DPNP_FN_COV
        DPNP_FN_DOT
        DPNP_FN_EIG
        DPNP_FN_FABS
        DPNP_FN_FLOOR
        DPNP_FN_FMOD
        DPNP_FN_MATMUL
        DPNP_FN_MAXIMUM
        DPNP_FN_MINIMUM
        DPNP_FN_PROD
        DPNP_FN_RAND
        DPNP_FN_SIGN
        DPNP_FN_SUM
        DPNP_FN_TRUNC


cdef extern from "backend_iface_fptr.hpp" namespace "DPNPFuncType":  # need this namespace for Enum import
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE

cdef extern from "backend_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void * ptr

    DPNPFuncData get_dpnp_function_ptr(DPNPFuncName name, DPNPFuncType first_type, DPNPFuncType second_type)


cdef DPNPFuncName get_DPNPFuncName_from_str(name):
    if name == "dpnp_dot":
        return DPNPFuncName.DPNP_FN_DOT
    elif name == "dpnp_matmul":
        return DPNPFuncName.DPNP_FN_MATMUL
    elif name == "dpnp_sum":
    elif name == "dpnp_sum":
        return DPNPFuncName.DPNP_FN_SUM
    elif name == "dpnp_argmax":
        return DPNPFuncName.DPNP_FN_ARGMAX
    elif name == "dpnp_argmin":
        return DPNPFuncName.DPNP_FN_ARGMIN
    elif name == "dpnp_argsort":
        return DPNPFuncName.DPNP_FN_ARGSORT
    elif name == "dpnp_cov":
        return DPNPFuncName.DPNP_FN_COV
    else:
        return DPNPFuncName.DPNP_FN_NONE


cdef DPNPFuncType get_DPNPFuncType_from_str(name):
    if name == "float32":
        return DPNPFuncType.DPNP_FT_FLOAT
    elif name == "int32":
        return DPNPFuncType.DPNP_FT_INT
    elif name == "float64":
        return DPNPFuncType.DPNP_FT_DOUBLE
    elif name == "int64":
        return DPNPFuncType.DPNP_FT_LONG
    else:
        return DPNPFuncType.DPNP_FT_NONE

from libc.stdio cimport printf
from libc.stdint cimport uintptr_t

cpdef get_dpnp_fn_ptr(name, types):
    cdef DPNPFuncName func_name = get_DPNPFuncName_from_str(name)
    cdef DPNPFuncType first_type = get_DPNPFuncType_from_str(types[0])
    cdef DPNPFuncType second_type = DPNP_FT_NONE

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(func_name, first_type, second_type)
    cdef uintptr_t fn_ptr = <uintptr_t>kernel_data.ptr

    return <object>fn_ptr
