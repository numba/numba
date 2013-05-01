# -*- coding: utf-8 -*-

"""
Universes of type constructors for numba.
"""

from __future__ import print_function, division, absolute_import

import struct as struct_
import ctypes

from numba.typesystem.typesystem import Universe
from numba.typesystem import kinds, types, llvmtyping

import llvm.core

import numpy as np

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

_plat_bits = struct_.calcsize('@P') * 8

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
    "short":        struct_.calcsize("h"),
    "int":          struct_.calcsize("i"),
    "long":         struct_.calcsize("l"),
    "longlong":     struct_.calcsize("Q"),
    "Py_ssize_t":   getsize('c_size_t', _plat_bits // 8),
    "npy_intp":     ctypes.sizeof(ctypes_npy_intp),
    # Unsigned int
    "ushort":       struct_.calcsize("H"),
    "uint":         struct_.calcsize("I"),
    "ulong":        struct_.calcsize("L"),
    "ulonglong":    struct_.calcsize("Q"),
    "size_t":       getsize('c_size_t', _plat_bits // 8),
    "Py_uintptr_t": ctypes.sizeof(ctypes.c_void_p),
    # Float
    "longdouble":   ctypes.sizeof(ctypes.c_longdouble),
    # Pointer
    "pointer":      ctypes.sizeof(ctypes.c_void_p),
}

default_type_sizes = dict(type_sizes, **native_sizes)

# ctors
mono = types.NumbaType.mono
poly = types.NumbaType.poly

def make_monotypes(kind_to_typename, monotypes):
    for kind, typenames in kind_to_typename.iteritems():
        for typename in typenames:
            monotypes[typename] = mono(kind, typename)

#------------------------------------------------------------------------
# Low-level universe
#------------------------------------------------------------------------

class LowLevelUniverse(Universe):
    """
    Type universe for low-level types:

        ints, floats, structs, pointers, functions, void
    """

    name = "low-level"

    kind_to_typenames = {
        # KIND -> [typename]
        kinds.KIND_INT: int_typenames,
        kinds.KIND_FLOAT: float_typenames,
        kinds.KIND_VOID: ["void"],
    }

    polytypes = {
        kinds.KIND_STRUCT: types.StructType,     # method 'struct'
        kinds.KIND_POINTER: types.PointerType,   # method 'pointer'
        kinds.KIND_FUNCTION: types.FunctionType, # method 'function'
    }

    def __init__(self, itemsizes=None):
        super(LowLevelUniverse, self).__init__(itemsizes or default_type_sizes)

    def make_monotypes(self, monotypes):
        make_monotypes(self.kind_to_typenames, monotypes)

    def rank(self, type):
        return 0 # TODO: implement

    def itemsize(self, type):
        if type.is_mono:
            assert type != self.void, "Void type has no itemsize"
            return self.itemsizes[type.typename]
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

    name = "llvm"

    polytypes = {
        kinds.KIND_STRUCT: llvm_poly(llvmtyping.lstruct),
        kinds.KIND_POINTER: llvm_poly(llvmtyping.lpointer),
        kinds.KIND_FUNCTION: llvm_poly(llvmtyping.lfunction),
    }

    def __init__(self, itemsizes=None):
        itemsizes = itemsizes or default_type_sizes
        super(LLVMUniverse, self).__init__(itemsizes)

    def make_monotypes(self, monotypes):
        lluniverse = LowLevelUniverse(itemsizes=self.itemsizes)
        size = lambda name: lluniverse.itemsize(getattr(lluniverse, name))

        for typename in int_typenames:
            monotypes[typename] = llvmtyping.lint(typename, size(typename))
        for typename in float_typenames:
            monotypes[typename] = llvmtyping.lfloat(typename, size(typename))

        monotypes["void"] = llvm.core.Type.void()

#------------------------------------------------------------------------
# Numba User-level Universe
#------------------------------------------------------------------------

numba_types = [
    "object", "null", "none", "ellipsis", "slice", "newaxis", "range",
]

def _struct(fields=(), name=None, readonly=False, packed=False, **kwargs):
    "Create a *mutable* struct type"
    if fields and kwargs:
        raise TypeError("The struct must be either ordered or unordered")
    elif kwargs:
        # fields = sort_types(kwargs)
        fields = list(kwargs.iteritems())

    return types.MutableStructType(fields, name, readonly, packed)

_ts = types.numba_registry.registry.items()

class NumbaUniverse(Universe):
    """
    Universe of numba types. Extends the types of the low-level universe.
    """

    lowlevel_universe = lowlevel_universe
    name = "numba"

    polytypes = {
        kinds.KIND_ARRAY: types.ArrayType,
        kinds.KIND_COMPLEX: types.ComplexType,
        kinds.KIND_POINTER: types.PointerType,
        kinds.KIND_FUNCTION: types.FunctionType,
        # ...
    }
    polytypes.update(
        [(name, ty) for name, (ty, mutable) in _ts if not mutable])

    mutable_polytypes = {
        kinds.KIND_STRUCT: _struct,
    }
    mutable_polytypes.update(
        [(name, ty) for name, (ty, mutable) in _ts if mutable])

    def __init__(self, *args, **kwargs):
        super(NumbaUniverse, self).__init__(*args, **kwargs)

        self.complex64 = self.complex(self.float)
        self.complex128 = self.complex(self.double)
        self.complex256 = self.complex(self.longdouble)

    def make_monotypes(self, monotypes):
        monotypes.update(self.lowlevel_universe.monotypes)
        for typename in numba_types:
            monotypes[typename] = mono(typename, typename)

    def typenames(self):
        return super(NumbaUniverse, self).typenames() + [
                 "complex64", "complex128", "complex256"]

    # def __getattr__(self, attr):
    #     return getattr(self.lowlevel_universe, attr)