# distutils: language = c++
# cython: language_level=3

import ctypes


cdef extern from "backend_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_NONE
        DPNP_FN_ADD
        DPNP_FN_ARGMAX
        DPNP_FN_ARGMIN
        DPNP_FN_DOT
        DPNP_FN_FABS
        DPNP_FN_MAXIMUM
        DPNP_FN_MINIMUM
        DPNP_FN_LAST

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
    elif name == "dpnp_sum":
        return DPNPFuncName.DPNP_FN_ADD
    elif name == "dpnp_argmax":
        return DPNPFuncName.DPNP_FN_ARGMAX
    elif name == "dpnp_argsort":
        return DPNPFuncName.DPNP_FN_ADD
    elif name == "dpnp_cov":
        return DPNPFuncName.DPNP_FN_ADD
    else:
        return DPNPFuncName.DPNP_FN_ADD


cdef DPNPFuncType get_DPNPFuncType_from_str(name):
    if name == "float32":
        return DPNPFuncType.DPNP_FT_FLOAT
    elif name == "int32":
        return DPNPFuncType.DPNP_FT_INT
    elif name == "float64":
        return DPNPFuncType.DPNP_FT_DOUBLE
    elif name == "int64":
        return DPNPFuncType.DPNP_FT_LONG

from libc.stdio cimport printf

cpdef get_dpnp_fn_ptr(name, types):
    print("inside function")
    cdef DPNPFuncName func_name = get_DPNPFuncName_from_str(name)
    cdef DPNPFuncType first_type = get_DPNPFuncType_from_str(types[0])
    cdef DPNPFuncType second_type = DPNP_FT_NONE

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(func_name, first_type, second_type)
    printf("pointer %p\n", kernel_data.ptr)
    print(hex(<object>kernel_data.ptr))
