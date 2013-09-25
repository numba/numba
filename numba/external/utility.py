# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

from numba import *
from numba.external import external
from numba.external.utilities import utilities
from numba.exttypes.virtual import PyCustomSlots_Table

class UtilityFunction(external.ExternalFunction):
    """
    A utility function written in a native language.

    funcaddr: the integer address of the C utility function
    See ExternalFunction for keyword arguments!
    """

    def __init__(self, funcaddr, return_type, arg_types, **kwargs):
        super(UtilityFunction, self).__init__(return_type, arg_types, **kwargs)
        self.funcaddr = funcaddr

    def declare_lfunc(self, context, llvm_module):
        lsig = self.signature.pointer().to_llvm(context)
        inttype = Py_uintptr_t.to_llvm(context)
        intval = llvm.core.Constant.int(inttype, self.funcaddr)
        lfunc = intval.inttoptr(lsig)
        return lfunc

    @classmethod
    def load(cls, func_name, signature, **kwds):
        """
        Load a utility function by name from the
        numba.external.utilities.utilities module.
        """
        # Get the integer address of C utility function
        func_addr = getattr(utilities, func_name)
        return cls(func_addr, signature.return_type, signature.args,
                   func_name=func_name, **kwds)


load = UtilityFunction.load
load2 = lambda name, sig: load(name, sig, check_pyerr_occurred=True)

object_to_numeric = {
    char       : load2("__Numba_PyInt_AsSignedChar", char(object_)),
    uchar      : load2("__Numba_PyInt_AsUnsignedChar", uchar(object_)),
    short      : load2("__Numba_PyInt_AsSignedShort", short(object_)),
    ushort     : load2("__Numba_PyInt_AsUnsignedShort", ushort(object_)),
    int_       : load2("__Numba_PyInt_AsSignedInt", int_(object_)),
    uint       : load2("__Numba_PyInt_AsUnsignedInt", uint(object_)),
    long_      : load2("__Numba_PyInt_AsSignedLong", long_(object_)),
    ulong      : load2("__Numba_PyInt_AsUnsignedLong", ulong(object_)),
    longlong   : load2("__Numba_PyInt_AsSignedLongLong", longlong(object_)),
    ulonglong  : load2("__Numba_PyInt_AsUnsignedLongLong", ulonglong(object_)),
}

void_p = void.pointer()
void_pp = void_p.pointer()

utility_funcs = list(object_to_numeric.itervalues()) + [
    UtilityFunction.load(
        "lookup_method", void_p(void_pp, uint64, char.pointer())),
    UtilityFunction.load(
        "Raise", int_(*[void_p] * 4),
        badval=-1,
    ),
    UtilityFunction.load("__Numba_PyInt_FromLongLong", object_(longlong)),
    UtilityFunction.load("__Numba_PyInt_FromUnsignedLongLong", object_(ulonglong)),

    UtilityFunction.load("convert_datetime_str_to_timestamp", int64(string_)),
    UtilityFunction.load("convert_datetime_str_to_units", int_(string_)),

    UtilityFunction.load("convert_numpy_datetime_to_timestamp", int64(object_)),
    UtilityFunction.load("convert_numpy_datetime_to_units", int_(object_)),

    UtilityFunction.load("convert_numpy_timedelta_to_diff", int64(object_)),
    UtilityFunction.load("convert_numpy_timedelta_to_units", int_(object_)),

    UtilityFunction.load("create_numpy_datetime",
        object_(int64, int_, object_)),
    UtilityFunction.load("create_numpy_timedelta",
        object_(int64, int_, object_)),

    UtilityFunction.load("get_target_unit_for_datetime_datetime",
        int_(int_, int_)),
    UtilityFunction.load("get_target_unit_for_timedelta_timedelta",
        int_(int_, int_)),
    UtilityFunction.load("get_target_unit_for_datetime_timedelta",
        int_(int_, int_)),

    UtilityFunction.load("extract_datetime_year", int64(int64, int32)),
    UtilityFunction.load("extract_datetime_month", int_(int64, int32)),
    UtilityFunction.load("extract_datetime_day", int_(int64, int32)),
    UtilityFunction.load("extract_datetime_hour", int_(int64, int32)),
    UtilityFunction.load("extract_datetime_min", int_(int64, int32)),
    UtilityFunction.load("extract_datetime_sec", int_(int64, int32)),

    UtilityFunction.load("sub_datetime_datetime",
        int64(int64, int_, int64, int_, int_)),
    UtilityFunction.load("add_datetime_timedelta",
        int64(int64, int_, int64, int_, int_)),
    UtilityFunction.load("sub_datetime_timedelta",
        int64(int64, int_, int64, int_, int_)),    
    UtilityFunction.load("extract_timedelta_sec", int_(int64, int32)),
    UtilityFunction.load("convert_timedelta_units_str", int_(string_)),
    UtilityFunction.load("get_units_num", int32(string_)),
]

def default_utility_library(context):
    """
    Create a library of utility functions.
    """
    extlib = external.ExternalLibrary(context)

    for utility_func in utility_funcs:
        extlib.add(utility_func)

    return extlib
