# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function, division, absolute_import

import math
import copy
import struct as struct_
import textwrap
import ctypes

from numba.typesystem.typesystem import (
    Universe, Type, MonoType, PolyType, Conser, nbo)
from numba.typesystem import typesystem

import numpy as np
import llvm.core


atom_type_names = [
    'Py_ssize_t', 'void', 'char', 'uchar', 'short', 'ushort',
    'int_', 'uint', 'long_', 'ulong', 'longlong', 'ulonglong',
    'size_t', 'npy_intp', 'c_string_type', 'bool_', 'object_',
    'float_', 'double', 'longdouble', 'float32', 'float64', 'float128',
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
    'complex64', 'complex128', 'complex256', 'struct', 'Py_uintptr_t'
]

__all__ = atom_type_names

#------------------------------------------------------------------------
# User-facing type functionality
#------------------------------------------------------------------------

def slice_type(ts, type, item):
    """
    Support array type creation by slicing, e.g. double[:, :] specifies
    a 2D strided array of doubles. The syntax is the same as for
    Cython memoryviews.
    """
    assert isinstance(item, (tuple, slice))

    def verify_slice(s):
        if s.start or s.stop or s.step not in (None, 1):
            raise ValueError(
                "Only a step of 1 may be provided to indicate C or "
                "Fortran contiguity")

    if isinstance(item, tuple):
        step_idx = None
        for idx, s in enumerate(item):
            verify_slice(s)
            if s.step and (step_idx or idx not in (0, len(item) - 1)):
                raise ValueError(
                    "Step may only be provided once, and only in the "
                    "first or last dimension.")

            if s.step == 1:
                step_idx = idx

        return ts.array(type, len(item),
                        is_c_contig=step_idx == len(item) - 1,
                        is_f_contig=step_idx == 0)
    else:
        verify_slice(item)
        return ts.array(type, 1, is_c_contig=bool(item.step))


def call_type(ts, type, *args):
    """
    Return a new function type when called with type arguments.
    """
    if len(args) == 1 and not isinstance(args[0], Type):
        # Cast in Python space
        # TODO: Create proxy object
        # TODO: Fully customizable type system (do this in Numba, not
        #       minivect)
        return args[0]

    return ts.function(type, args)

# ______________________________________________________________________
# Type methods

default_type_behaviour = {
    "__getitem__":  lambda self, item: slice_type(self.ts, self, item),
    "__call__":     lambda self, *args: call_type(self.ts, self, *args),
    "pointer":      lambda self: self.ts.pointer(self),
    # 'context' is for backwards compatibility
    "to_llvm":      lambda self, context: self.ts.convert("llvm", self),
    "to_ctypes":    lambda self: self.ts.convert("ctypes", self),
    "get_dtype":    lambda self: self.ts.convert("numpy", self),
}

def annotate_type(cls):
    for name, meth in default_type_behaviour.iteritems():
        setattr(cls, name, meth)

def make_polytype(typename, names):
    """
    Create a new polytype that has named attributes. E.g.

        make_polytype("ArrayType", ["dtype", "ndim"])
    """
    # Create parameter accessors
    typedict = dict([(name, lambda self, i=i: self.params[i])
                        for i, name in enumerate(names)])
    return type(typename, (PolyType,), typedict)


@annotate_type
class NumbaMonoType(MonoType):
    """
    MonoType with user-facing methods:

        call: create a function type
        slice: create an array type
        conversion: to_llvm/to_ctypes/get_dtype
    """

@annotate_type
class NumbaPolyType(PolyType):
    """
    PolyType with user-facing methods:

        call: create a function type
        slice: create an array type
        conversion: to_llvm/to_ctypes/get_dtype
    """

#------------------------------------------------------------------------
# Type constructors
#------------------------------------------------------------------------

class ArrayType(NumbaPolyType):
    """
    An array type. ArrayType may be sliced to obtain a subtype:

    >>> double[:, :, ::1][1:]
    double[:, ::1]
    >>> double[:, :, ::1][:-1]
    double[:, :]

    >>> double[::1, :, :][:-1]
    double[::1, :]
    >>> double[::1, :, :][1:]
    double[:, :]
    """

    is_array = True

    def __init__(self, ts, kind, dtype, ndim,
                 is_c_contig, is_f_contig, inner_contig):
        super(ArrayType, self).__init__(ts, kind)

        assert dtype is not None
        self.dtype = dtype
        self.ndim = ndim

        # Flags
        if ndim == 1 and (is_c_contig or is_f_contig):
            self.is_c_contig = True
            self.is_f_contig = True

        self.inner_contig = inner_contig or is_c_contig or is_f_contig

    def pointer(self):
        raise Exception("You probably want a pointer type to the dtype")

    def __repr__(self):
        axes = [":"] * self.ndim
        if self.is_c_contig and self.ndim > 0:
            axes[-1] = "::1"
        elif self.is_f_contig and self.ndim > 0:
            axes[0] = "::1"

        return "%s[%s]" % (self.dtype, ", ".join(axes))

    def copy(self, **kwargs):
        if 'dtype' in kwargs:
            assert kwargs['dtype'] is not None
        array_type = copy.copy(self)
        vars(array_type).update(kwargs)
        return array_type

    @property
    def strided(self):
        type = self.copy()
        type.is_c_contig = False
        type.is_f_contig = False
        type.inner_contig = False
        type.broadcasting = None
        return type

    def __getitem__(self, index):
        "Slicing an array slices the dimensions"
        assert isinstance(index, slice)
        assert index.step is None
        assert index.start is not None or index.stop is not None

        start = 0
        stop = self.ndim
        if index.start is not None:
            start = index.start
        if index.stop is not None:
            stop = index.stop

        ndim = len(range(self.ndim)[start:stop])

        if ndim == 0:
            type = self.dtype
        elif ndim > 0:
            type = self.strided
            type.ndim = ndim
            type.is_c_contig = self.is_c_contig and stop == self.ndim
            type.is_f_contig = self.is_f_contig and start == 0
            type.inner_contig = type.is_c_contig or type.is_f_contig
            if type.broadcasting:
                type.broadcasting = self.broadcasting[start:stop]
        else:
            raise IndexError(index, ndim)

        return type


PointerType = make_polytype("PointerType", ["base_type"])
CArrayType = make_polytype("CArrayType", ["base_type"])



class FloatType(NumericType):
    is_float = True

    kind = FLOAT_KIND

    def declare(self):
        if self.itemsize == 4:
            return "float"
        elif self.itemsize == 8:
            return "double"
        else:
            return str(self)

    @property
    def comparison_type_list(self):
        return self.subtype_list + [self.itemsize]

    def __eq__(self, other):
        return isinstance(other, FloatType) and self.itemsize == other.itemsize

    __hash__ = NumericType.__hash__



class ComplexType(NumericType):
    is_complex = True
    subtypes = ['base_type']

    kind = COMPLEX_KIND

class Py_ssize_t_Type(IntType):
    is_py_ssize_t = True
    name = "Py_ssize_t"
    rank = 9
    signed = True

    def __init__(self, **kwds):
        super(Py_ssize_t_Type, self).__init__(**kwds)
        if have_ctypes:
            if hasattr(ctypes, 'c_ssize_t'):
                self.itemsize = ctypes.sizeof(ctypes.c_ssize_t)
            else:
                self.itemsize = size_t.itemsize
        else:
            self.itemsize = _plat_bits // 8


class NPyIntp(IntType):
    is_numpy_intp = True
    name = "npy_intp"
    rank = 10

    def __init__(self, **kwds):
        super(NPyIntp, self).__init__(**kwds)
        ctypes_array = np.empty(0).ctypes.strides
        self.itemsize = ctypes.sizeof(ctypes_array._type_)

class CharType(IntType):
    is_char = True
    name = "char"
    rank = 1
    signed = True

    def to_llvm(self, context):
        return lc.Type.int(8)

class CStringType(Type):
    is_c_string = True

    def __repr__(self):
        return "const char *"

    def to_llvm(self, context):
        return char.pointer().to_llvm(context)

class VoidType(NamedType):
    is_void = True
    name = "void"

    def to_llvm(self, context):
        return lc.Type.void()

class ObjectType(Type):
    is_object = True
    itemsize = VoidType().pointer().itemsize

    def __repr__(self):
        return "PyObject *"

def pass_by_ref(type):
    return type.is_struct or type.is_complex

class Function(object):
    """
    Function types may be called with Python functions to create a Function
    object. This may be used to minivect users for their own purposes. e.g.

    @double(double, double)
    def myfunc(...):
       ...
    """
    def __init__(self, signature, py_func):
        self.signature = signature
        self.py_func = py_func

    def __call__(self, *args, **kwargs):
        """
        Implement this to pass the callable test for classmethod/staticmethod.
        E.g.

            @classmethod
            @void()
            def m(self):
                ...
        """
        raise TypeError("Not a callable function")

class FunctionType(Type):
    subtypes = ['return_type', 'args']
    is_function = True
    is_vararg = False

    struct_by_reference = False

    def __init__(self, return_type, args, name=None, is_vararg=False, **kwds):
        super(FunctionType, self).__init__(**kwds)
        self.return_type = return_type
        self.args = tuple(args)
        self.name = name
        self.is_vararg = is_vararg

    def __repr__(self):
        args = [str(arg) for arg in self.args]
        if self.is_vararg:
            args.append("...")
        if self.name:
            namestr = self.name
        else:
            namestr = ''

        return "%s (*%s)(%s)" % (self.return_type, namestr, ", ".join(args))

    @property
    def actual_signature(self):
        """
        Passing structs by value is not properly supported for different
        calling conventions in LLVM, so we take an extra argument
        pointing to a caller-allocated struct value.
        """
        if self.struct_by_reference:
            args = []
            for arg in self.args:
                if pass_by_ref(arg):
                    arg = arg.pointer()
                args.append(arg)

            return_type = self.return_type
            if pass_by_ref(self.return_type):
                return_type = void
                args.append(self.return_type.pointer())

            self = FunctionType(return_type, args)

        return self

    @property
    def struct_return_type(self):
        # Function returns a struct.
        return self.return_type.pointer()

    def __call__(self, *args):
        if len(args) != 1 or isinstance(args[0], Type):
            return super(FunctionType, self).__call__(*args)

        assert self.return_type is not None
        assert self.args is not None
        func, = args
        return Function(self, func)



def _sort_types_key(field_type):
    if field_type.is_complex:
        return field_type.base_type.rank * 2
    elif field_type.is_numeric or field_type.is_struct:
        return field_type.rank
    elif field_type.is_vector:
        return _sort_types_key(field_type.element_type) * field_type.vector_size
    elif field_type.is_carray:
        return _sort_types_key(field_type.base_type) * field_type.size
    elif field_type.is_pointer or field_type.is_object or field_type.is_array:
        return 8
    else:
        return 1

def _sort_key(keyvalue):
    field_name, field_type = keyvalue
    return _sort_types_key(field_type)

def sort_types(types_dict):
    # reverse sort on rank, forward sort on name
    d = {}
    for field in types_dict.iteritems():
        key = _sort_key(field)
        d.setdefault(key, []).append(field)

    def key(keyvalue):
        field_name, field_type = keyvalue
        return field_name

    fields = []
    for rank in sorted(d, reverse=True):
        fields.extend(sorted(d[rank], key=key))

    return fields

class StructType(Type):
    """
    Create a struct type. Fields may be ordered or unordered. Unordered fields
    will be ordered from big types to small types (for better alignment).

    >>> struct([('a', int_), ('b', float_)], name='Foo') # ordered struct
    struct Foo { int a, float b }
    >>> struct(a=int_, b=float_, name='Foo') # unordered struct
    struct Foo { float b, int a }
    >>> struct(a=int32, b=int32, name='Foo') # unordered struct
    struct Foo { int32 a, int32 b }

    >>> S = struct(a=complex128, b=complex64, c=struct(f1=double, f2=double, f3=int32))
    >>> S
    struct { struct { double f1, double f2, int32 f3 } c, complex128 a, complex64 b }

    >>> S.offsetof('a')
    24
    """

    is_struct = True

    def __init__(self, fields=(), name=None,
                 readonly=False, packed=False, **kwargs):
        super(StructType, self).__init__()
        if fields and kwargs:
            raise TypeError("The struct must be either ordered or unordered")

        if kwargs:
            fields = sort_types(kwargs)

        self.fields = list(fields)
        self.name = name
        self.readonly = readonly
        self.fielddict = dict(self.fields)
        self.packed = packed

        self.update_mutated()

    def copy(self):
        return self.ts.struct(self.fields, self.name, self.readonly, self.packed)

    def __repr__(self):
        if self.name:
            name = self.name + ' '
        else:
            name = ''
        return 'struct %s{ %s }' % (
                name, ", ".join("%s %s" % (field_type, field_name)
                                    for field_name, field_type in self.fields))

    def __eq__(self, other):
        return other.is_struct and self.fields == other.fields

    def __hash__(self):
        return hash(tuple(self.fields))

    def is_prefix(self, other_struct):
        other_fields = other_struct.fields[:len(self.fields)]
        return self.fields == other_fields

    def add_field(self, name, type):
        assert name not in self.fielddict
        self.fielddict[name] = type
        self.fields.append((name, type))
        self.mutated = True

    def update_mutated(self):
        self.rank = sum([_sort_key(field) for field in self.fields])
        self.mutated = False

    def offsetof(self, field_name):
        """
        Compute the offset of a field. Must be used only after mutation has
        finished.
        """
        ctype = self.to_ctypes()
        return getattr(ctype, field_name).offset


def getsize(ctypes_name, default):
    try:
        return ctypes.sizeof(getattr(ctypes, ctypes_name))
    except ImportError:
        return default

def get_target_triple():
    target_machine = llvm.ee.TargetMachine.new()
    is_ppc = target_machine.triple.startswith("ppc")
    is_x86 = target_machine.triple.startswith("x86")
    return is_ppc, is_x86

#
### Internal types
#
c_string_type = CStringType()
void = VoidType()

#
### Public types
#
try:
    npy_intp = NPyIntp()
except ImportError:
    npy_intp = None

size_t = IntType(name="size_t", rank=8.5,
                 itemsize=getsize('c_size_t', _plat_bits // 8), signed=False)
Py_ssize_t = Py_ssize_t_Type()
Py_uintptr_t = IntType(name='Py_uintptr_t',
                       itemsize=getsize('c_void_p', Py_ssize_t.itemsize),
                       rank=8.5)

char = CharType(name="char", typecode='b')
short = IntType(name="short", rank=2, typecode='h')
int_ = IntType(name="int", rank=4, typecode='i')
long_ = IntType(name="long", rank=5, typecode='l')
longlong = IntType(name="PY_LONG_LONG", rank=8, typecode='q')

uchar = CharType(name="unsigned char", signed=False, typecode='B')
ushort = IntType(name="unsigned short", rank=2.5,
                 typecode='H', signed=False)
uint = IntType(name="unsigned int", rank=4.5, typecode='I', signed=False)
ulong = IntType(name="unsigned long", rank=5.5, typecode='L', signed=False)
ulonglong = IntType(name="unsigned PY_LONG_LONG", rank=8.5,
                    typecode='Q', signed=False)

float_ = FloatType(name="float", rank=20, itemsize=4)
double = FloatType(name="double", rank=21, itemsize=8)
longdouble = FloatType(name="long double", rank=22,
                       itemsize=ctypes.sizeof(ctypes.c_longdouble))

bool_ = BoolType()
object_ = ObjectType()

int8 = IntType(name="int8", rank=1, itemsize=1)
int16 = IntType(name="int16", rank=2, itemsize=2)
int32 = IntType(name="int32", rank=4, itemsize=4)
int64 = IntType(name="int64", rank=8, itemsize=8)

uint8 = IntType(name="uint8", rank=1.5, signed=False, itemsize=1)
uint16 = IntType(name="uint16", rank=2.5, signed=False, itemsize=2)
uint32 = IntType(name="uint32", rank=4.5, signed=False, itemsize=4)
uint64 = IntType(name="uint64", rank=8.5, signed=False, itemsize=8)

float32 = FloatType(name="float32", rank=20, itemsize=4)
float64 = FloatType(name="float64", rank=21, itemsize=8)
float128 = FloatType(name="float128", rank=22, itemsize=16)

complex64 = ComplexType(name="complex64", base_type=float32,
                        rank=30, itemsize=8)
complex128 = ComplexType(name="complex128", base_type=float64,
                         rank=31, itemsize=16)
complex256 = ComplexType(name="complex256", base_type=float128,
                         rank=32, itemsize=32)

integral = []
native_integral = []
floating = []
complextypes = []

for typename in __all__:
    minitype = globals()[typename]
    if minitype is None:
        continue

    if minitype.is_int:
        integral.append(minitype)
    elif minitype.is_float:
        floating.append(minitype)
    elif minitype.is_complex:
        complextypes.append(minitype)

numeric = integral + floating + complextypes
native_integral.extend((Py_ssize_t, size_t))

integral.sort(key=_sort_types_key)
native_integral = [minitype for minitype in integral
                                if minitype.typecode is not None]
floating.sort(key=_sort_types_key)
complextypes.sort(key=_sort_types_key)



def find_type_of_size(size, typelist):
    for type in typelist:
        if type.itemsize == size:
            return type

    assert False, "Type of size %d not found: %s" % (size, typelist)


class DefaultConstantTyper(object):

    def typeof(self, ts, value):
        u = ts.universe

        if isinstance(value, float):
            return ts.universe.double
        elif isinstance(value, bool):
            return ts.universe.bool
        elif isinstance(value, (int, long)):
            if abs(value) < 1:
                bits = 0
            else:
                bits = math.ceil(math.log(abs(value), 2))

            if bits < 32:
                return int_
            elif bits < 64:
                return int64
            else:
                raise ValueError("Cannot represent %s as int32 or int64", value)
        elif isinstance(value, complex):
            return complex128
        elif isinstance(value, str):
            return c_string_type
        elif isinstance(value, np.ndarray):
            dtype = map_dtype(value.dtype)
            return ArrayType(dtype, value.ndim,
                             is_c_contig=value.flags['C_CONTIGUOUS'],
                             is_f_contig=value.flags['F_CONTIGUOUS'])
        else:
            return object_


class NumbaUniverse(Universe):

    # typename -> type
    ints = {}
    uints = {}
    floats = {}
    complexes = {}

    functions = Conser()
    arrays = Conser()
    pointers = Conser()
    carrays = Conser()

    def init(self, ts):
        for name in atom_type_names:
            type = MonoType(ts, kind, name)
            setattr(self, name, type)
            self.monotypes[name] = type

    function = FunctionType
    array = ArrayType
    pointer = PointerType
    carray = CArrayType
    struct = struct


class AtomUnification(object):

    def promote_numeric(self, type1, type2):
        "Promote two numeric types"
        type = max([type1, type2], key=lambda type: type.rank)
        if type1.kind != type2.kind:
            def itemsize(type):
                return type.itemsize // 2 if type.is_complex else type.itemsize

            size = max(itemsize(type1), itemsize(type2))
            if type.is_complex:
                type = find_type_of_size(size * 2, complextypes)
            elif type.is_float:
                type = find_type_of_size(size, floating)
            else:
                assert type.is_int
                type = find_type_of_size(size, integral)

        return type

    def promote_arrays(self, type1, type2):
        "Promote two array types in an expression to a new array type"
        equal_ndim = type1.ndim == type2.ndim
        return ArrayType(self.unify(type1.dtype, type2.dtype),
                         ndim=max((type1.ndim, type2.ndim)),
                         is_c_contig=(equal_ndim and type1.is_c_contig and
                                      type2.is_c_contig),
                         is_f_contig=(equal_ndim and type1.is_f_contig and
                                      type2.is_f_contig))

    def unify(self, type1, type2):
        "Promote two arbitrary types"
        string_types = c_string_type, char.pointer()
        if type1.is_pointer and type2.is_int_like:
            return type1
        elif type2.is_pointer and type2.is_int_like:
            return type2
        elif type1.is_object or type2.is_object:
            return object_
        elif type1.is_numeric and type2.is_numeric:
            return self.promote_numeric(type1, type2)
        elif type1.is_array and type2.is_array:
            return self.promote_arrays(type1, type2)
        elif type1 in string_types and type2 in string_types:
            return c_string_type
        elif type1.is_bool and type2.is_bool:
            return bool_
        else:
            raise minierror.UnpromotableTypeError((type1, type2))


def get_utility():
    import numpy

    return textwrap.dedent("""\
    #include <stdint.h>

    #ifndef HAVE_LONGDOUBLE
        #define HAVE_LONGDOUBLE %d
    #endif

    typedef struct {
        float real;
        float imag;
    } complex64;

    typedef struct {
        double real;
        double imag;
    } complex128;

    #if HAVE_LONGDOUBLE
    typedef struct {
        long double real;
        long double imag;
    } complex256;
    #endif

    typedef float float32;
    typedef double float64;
    #if HAVE_LONGDOUBLE
    typedef long double float128;
    #endif
    """ % hasattr(numpy, 'complex256'))

if __name__ == '__main__':
    import doctest
    doctest.testmod()