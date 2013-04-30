# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function, division, absolute_import

import math
import copy
import struct
import ctypes
import textwrap
from functools import partial

from numba.typesystem.typesystem import (
    Universe, Type, Conser, nbo)
from numba.typesystem import typesystem
from numba.typesystem.usertypes import *
from numba.typesystem.kinds import *
from numba.typesystem import llvmtyping


int_typenames = [
    'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong',
    'longlong', 'ulonglong', 'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'size_t', 'npy_intp', 'Py_ssize_t', 'Py_uintptr_t',
    'bool', # hmm
]

float_typenames = [
    'float', 'double', 'longdouble', 'float32', 'float64', 'float128',
]

complex_typenames = [
    'complex64', 'complex128', 'complex256',
]

#------------------------------------------------------------------------
# Default type sizes
#------------------------------------------------------------------------

_plat_bits = struct.calcsize('@P') * 8

def getsize(ctypes_name, default):
    try:
        return ctypes.sizeof(getattr(ctypes, ctypes_name))
    except ImportError:
        return default

# Type sizes in bytes
type_sizes = {
    "bool":         1,
    # Int
    "char":         1,
    "int8":         1,
    "int16":        2,
    "int32":        4,
    "int64":        8,
    # Unsigned int
    "uchar":        1,
    "uint8":        1,
    "uint16":       2,
    "uint32":       4,
    "uint64":       8,
    # Float
    # "float16":      2,
    "float32":      4,
    "float64":      8,
    "float128":     16,
    "float":        4,
    "double":       8,
    # Complex
    "complex64":    8,
    "complex128":   16,
    "complex256":   32,
}

ctypes_npy_intp = np.empty(0).ctypes.strides._type_

native_sizes = {
    # Int
    "short":        struct.calcsize("h"),
    "int":          struct.calcsize("i"),
    "long":         struct.calcsize("l"),
    "longlong":     struct.calcsize("Q"),
    "Py_ssize_t":   getsize('c_size_t', _plat_bits // 8),
    "npy_intp":     ctypes.sizeof(ctypes_npy_intp),
    # Unsigned int
    "ushort":       struct.calcsize("H"),
    "uint":         struct.calcsize("I"),
    "ulong":        struct.calcsize("L"),
    "ulonglong":    struct.calcsize("Q"),
    "size_t":       getsize('c_size_t', _plat_bits // 8),
    "Py_uintptr_t": ctypes.sizeof(ctypes.c_void_p),
    # Float
    "longdouble":   ctypes.sizeof(ctypes.c_longdouble),
    # Pointer
    "pointer":      ctypes.sizeof(ctypes.c_void_p),
}

default_type_sizes = dict(type_sizes, **native_sizes)

# ctors
mono = NumbaType.mono
poly = NumbaType.poly

#------------------------------------------------------------------------
# Low-level universe
#------------------------------------------------------------------------

class LowLevelUniverse(Universe):
    """
    Type universe for low-level types:

        ints, floats, structs, pointers, functions, void
    """

    monokind_to_typenames = {
        # KIND -> [typename]
        KIND_INT: int_typenames,
        KIND_FLOAT: float_typenames,
        KIND_VOID: ["void"],
    }

    polytypes = {
        # KIND_STRUCT: StructType,   # This is a mutable type, don't cons
        KIND_POINTER: PointerType,   # method 'pointer'
        KIND_FUNCTION: FunctionType, # method 'function'
    }

    def __init__(self, kind_sorting=None, itemsizes=None):
        super(LowLevelUniverse, self).__init__(kind_sorting,
                                               itemsizes or default_type_sizes)

    def make_monotypes(self, monotypes):
        for kind, typenames in self.monokind_to_typenames.iteritems():
            for typename in typenames:
                monotypes[typename] = mono(kind, typename)

    def struct(self, fields=(), name=None, readonly=False, packed=False, **kwargs):
        if fields and kwargs:
            raise TypeError("The struct must be either ordered or unordered")
        elif kwargs:
            fields = sort_types(kwargs)

        return StructType(fields, name, readonly, packed)

    def itemsize(self, type):
        if type.is_mono:
            assert type != self.void, "Void type has no itemsize"
            return self.itemsizes[type.name]
        elif self.kind(type) in self.itemsizes:
            assert not type.is_mono
            return self.itemsizes[self.kind(type)]
        elif type.is_function:
            return self.itemsizes[KIND_POINTER]
        elif type.is_struct:
            return sum([self.itemsize(t) for n, t in type.fields])
        else:
            raise ValueError("Type %s has no itemsize" % (type,))

lowlevel_universe = LowLevelUniverse()

#------------------------------------------------------------------------
# LLVM Universe
#------------------------------------------------------------------------

def llvm_poly(llvm_ctor):
    def ctor(*params):
        return llvm_ctor(*params)
    return ctor

class LLVMUniverse(Universe):

    polytypes = {
        KIND_STRUCT: llvm_poly(llvmtyping.struct),
        KIND_POINTER: llvm_poly(llvmtyping.pointer),
        KIND_FUNCTION: llvm_poly(llvmtyping.function),
    }

    def __init__(self, kind_sorting=None, itemsizes=None):
        itemsizes = itemsizes or default_type_sizes
        self.lluniverse = LowLevelUniverse(kind_sorting, itemsizes=itemsizes)
        super(LLVMUniverse, self).__init__(kind_sorting, itemsizes)

    def make_monotypes(self, monotypes):
        itemsize = lambda name: self.lluniverse.itemsize(
            getattr(self.lluniverse, name))

        for typename in int_typenames:
            monotypes[typename] = llvm.core.Type.int(itemsize(typename) * 8)
        for typename in float_typenames:
            monotypes[typename] = llvmtyping.float(itemsize(typename))
        monotypes["void"] = llvm.core.Type.void()

#------------------------------------------------------------------------
# Numba User-level Universe
#------------------------------------------------------------------------

class NumbaUniverse(Universe):

    lowlevel_universe = lowlevel_universe

    def __init__(self, *args, **kwargs):
        super(NumbaUniverse, self).__init__(*args, **kwargs)
        self.polytypes.update(self.lowlevel_universe.polytypes)

    def make_monotypes(self, monotypes):
        monotypes.update(self.lowlevel_universe.monotypes)
        for name in complex_typenames:
            setattr(self, name, mono(KIND_COMPLEX, name))

    def __getattr__(self, attr):
        return getattr(self.lowlevel_universe, attr)