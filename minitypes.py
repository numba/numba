"""
This module provides a minimal type system, and ways to promote types, as
well as ways to convert to an LLVM type system. A set of predefined types are
defined. Types may be sliced to turn them into array types, in the same way
as the memoryview syntax.

>>> char
char
>>> int8[:, :, :]
int8[:, :, :]
>>> int8.signed
True
>>> uint8
uint8
>>> uint8.signed
False

>>> char.pointer()
char *
>>> int_[:, ::1]
int[:, ::1]
>>> int_[::1, :]
int[::1, :]
>>> double[:, ::1, :]
Traceback (most recent call last):
   ...
InvalidTypeSpecification: Step may only be provided once, and only in the first or last dimension.
"""

__all__ = ['Py_ssize_t', 'void', 'char', 'uchar', 'short', 'ushort',
           'int_', 'uint', 'long_', 'ulong', 'longlong', 'ulonglong',
           'size_t', 'npy_intp', 'c_string_type', 'bool_', 'object_',
           'float_', 'double', 'longdouble', 'float32', 'float64', 'float128',
           'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
           'complex64', 'complex128', 'complex256', 'struct']

import sys
import math
import copy
import struct as struct_
import ctypes
import textwrap

try:
    import llvm.core
    from llvm import core as lc
except ImportError:
    llvm = None

import miniutils
import minierror

# Check below taken from Numba
if sys.maxint > 2**33:
    _plat_bits = 64
else:
    _plat_bits = 32

if struct_.pack('i', 1)[0] == '\1':
    nbo = '<' # little endian
else:
    nbo = '>' # big endian

class TypeMapper(object):
    """
    Maps foreign types to minitypes. Users of minivect should implement
    this and pass it to :py:class:`minivect.miniast.Context`.

    >>> import miniast
    >>> context = miniast.Context()
    >>> miniast.typemapper = TypeMapper(context)
    >>> tm = context.typemapper

    >>> tm.promote_types(int8, double)
    double
    >>> tm.promote_types(int8, uint8)
    uint8
    >>> tm.promote_types(int8, complex128)
    complex128
    >>> tm.promote_types(int8, object_)
    PyObject *

    >>> tm.promote_types(int64, float32)
    float
    >>> tm.promote_types(int64, complex64)
    complex64
    >>> tm.promote_types(float32, float64)
    double
    >>> tm.promote_types(float32, complex64)
    complex64
    >>> tm.promote_types(complex64, complex128)
    complex128
    >>> tm.promote_types(complex256, object_)
    PyObject *

    >>> tm.promote_types(float32.pointer(), Py_ssize_t)
    float *
    >>> tm.promote_types(float32.pointer(), Py_ssize_t)
    float *
    >>> tm.promote_types(float32.pointer(), uint8)
    float *

    >>> tm.promote_types(float32.pointer(), float64.pointer())
    Traceback (most recent call last):
        ...
    UnpromotableTypeError: (float *, double *)

    >>> tm.promote_types(float32[:, ::1], float32[:, ::1])
    float[:, ::1]
    >>> tm.promote_types(float32[:, ::1], float64[:, ::1])
    double[:, ::1]
    >>> tm.promote_types(float32[:, ::1], float64[::1, :])
    double[:, :]
    >>> tm.promote_types(float32[:, :], complex128[:, :])
    complex128[:, :]
    >>> tm.promote_types(int_[:, :], object_[:, ::1])
    PyObject *[:, :]
    """

    def __init__(self, context):
        self.context = context

    def map_type(self, opaque_type):
        "Map a foreign type to a minitype"
        if opaque_type.is_int:
            return int_
        elif opaque_type.is_float:
            return float_
        elif opaque_type.is_double:
            return double
        elif opaque_type.is_pointer:
            return PointerType(self.map_type(opaque_type.base_type))
        elif opaque_type.is_py_ssize_t:
            return Py_ssize_t
        elif opaque_type.is_char:
            return char
        else:
            raise minierror.UnmappableTypeError(opaque_type)

    def to_llvm(self, type):
        "Return an LLVM type for the given type."
        raise NotImplementedError

    def from_python(self, value):
        "Get a type from a python value"
        np = sys.modules.get('numpy', None)

        if isinstance(value, float):
            return double
        elif isinstance(value, (int, long)):
            return int_
        elif isinstance(value, complex):
            return complex128
        elif isinstance(value, str):
            return c_string_type
        elif np and isinstance(value, np.ndarray):
            dtype = map_dtype(value.dtype)
            return ArrayType(dtype, value.ndim,
                             is_c_contig=value.flags['C_CONTIGUOUS'],
                             is_f_contig=value.flags['F_CONTIGUOUS'])
        else:
            return object_
            # raise minierror.UnmappableTypeError(type(value))

    def promote_numeric(self, type1, type2):
        "Promote two numeric types"
        return max([type1, type2], key=lambda type: type.rank)

    def promote_arrays(self, type1, type2):
        "Promote two array types in an expression to a new array type"
        equal_ndim = type1.ndim == type2.ndim
        return ArrayType(self.promote_types(type1.dtype, type2.dtype),
                         ndim=max(type1.ndim, type2.ndim),
                         is_c_contig=(equal_ndim and type1.is_c_contig and
                                      type2.is_c_contig),
                         is_f_contig=(equal_ndim and type1.is_f_contig and
                                      type2.is_f_contig))

    def promote_types(self, type1, type2):
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
        else:
            raise minierror.UnpromotableTypeError((type1, type2))

def map_dtype(dtype):
    """
    Map a NumPy dtype to a minitype.

    >>> import numpy as np
    >>> map_dtype(np.dtype(np.int32))
    int32
    >>> map_dtype(np.dtype(np.int64))
    int64
    >>> map_dtype(np.dtype(np.object))
    PyObject *
    >>> map_dtype(np.dtype(np.float64))
    double
    >>> map_dtype(np.dtype(np.complex128))
    complex128
    """
    import numpy as np

    if dtype.byteorder not in ('=', nbo) and dtype.kind in ('iufbc'):
        raise minierror.UnmappableTypeError(
                "Only native byteorder is supported", dtype)

    item_idx = int(math.log(dtype.itemsize, 2))
    if dtype.kind == 'i':
        return [int8, int16, int32, int64][item_idx]
    elif dtype.kind == 'u':
        return [uint8, uint16, uint32, uint64][item_idx]
    elif dtype.kind == 'f':
        if dtype.itemsize == 2:
            pass # half floats not supported yet
        elif dtype.itemsize == 4:
            return float32
        elif dtype.itemsize == 8:
            return float64
        elif dtype.itemsize == 16:
            return float128
    elif dtype.kind == 'b':
        return int8
    elif dtype.kind == 'c':
        if dtype.itemsize == 8:
            return complex64
        elif dtype.itemsize == 16:
            return complex128
        elif dtype.itemsize == 32:
            return complex256
    elif dtype.kind == 'V':
        fields = [(name, map_dtype(dtype.fields[name][0]))
                      for name in dtype.names]
        return struct(fields, packed=not dtype.isalignedstruct)
    elif dtype.kind == 'O':
        return object_

def create_dtypes():
    import numpy as np

    minitype2dtype = {
        int8     : np.int8,
        int16    : np.int16,
        int32    : np.int32,
        int64    : np.int64,
        uint8    : np.uint8,
        uint16   : np.uint16,
        uint32   : np.uint32,
        uint64   : np.uint64,

        float_   : np.float32,
        double   : np.float64,
        longdouble: np.longdouble,

        short    : np.dtype('h'),
        int_     : np.dtype('i'),
        long_    : np.dtype('l'),
        longlong : np.longlong,
        ushort   : np.dtype('H'),
        uint     : np.dtype('I'),
        ulong    : np.dtype('L'),
        ulonglong: np.ulonglong,

        complex64: np.complex64,
        complex128: np.complex128,
        complex256: getattr(np, 'complex256', None),

        object_  : np.object,
    }

    return dict((k, np.dtype(v)) for k, v in minitype2dtype.iteritems())

_dtypes = None
def map_minitype_to_dtype(type):
    global _dtypes

    if type.is_struct:
        import numpy as np

        fields = [(field_name, map_minitype_to_dtype(field_type))
                      for field_name, field_type in type.fields]
        return np.dtype(fields, align=not type.packed)

    if _dtypes is None:
        _dtypes = create_dtypes()

    if type.is_array:
        type = type.dtype

    dtype = _dtypes[type]
    assert dtype is not None, "dtype not supported in this numpy build"
    return dtype

NONE_KIND = 0
INT_KIND = 1
FLOAT_KIND = 2
COMPLEX_KIND = 3

class Type(miniutils.ComparableObjectMixin):
    """
    Base class for all types.

    .. attribute:: subtypes

        The list of subtypes to allow comparing and hashing them recursively
    """

    is_array = False
    is_pointer = False
    is_typewrapper = False

    is_bool = False
    is_numeric = False
    is_py_ssize_t = False
    is_char = False
    is_int = False
    is_float = False
    is_c_string = False
    is_object = False
    is_function = False
    is_int_like = False
    is_complex = False
    is_void = False

    kind = NONE_KIND

    subtypes = []

    def __init__(self, **kwds):
        vars(self).update(kwds)
        self.qualifiers = kwds.get('qualifiers', frozenset())

    def qualify(self, *qualifiers):
        "Qualify this type with a qualifier such as ``const`` or ``restrict``"
        qualifiers = list(qualifiers)
        qualifiers.extend(self.qualifiers)
        attribs = dict(vars(self), qualifiers=qualifiers)
        return type(self)(**attribs)

    def unqualify(self, *unqualifiers):
        "Remove the given qualifiers from the type"
        unqualifiers = set(unqualifiers)
        qualifiers = [q for q in self.qualifiers if q not in unqualifiers]
        attribs = dict(vars(self), qualifiers=qualifiers)
        return type(self)(**attribs)

    def pointer(self):
        "Get a pointer to this type"
        return PointerType(self)

    @property
    def subtype_list(self):
        return [getattr(self, subtype) for subtype in self.subtypes]

    @property
    def comparison_type_list(self):
        return self.subtype_list

    def __eq__(self, other):
        # Don't use isinstance here, compare on exact type to be consistent
        # with __hash__. Override where sensible
        return (type(self) is type(other) and
                self.comparison_type_list == other.comparison_type_list)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        h = hash(type(self))
        for subtype in self.comparison_type_list:
            h = h ^ hash(subtype)

        return h

    def __getitem__(self, item):
        """
        Support array type creation by slicing, e.g. double[:, :] specifies
        a 2D strided array of doubles. The syntax is the same as for
        Cython memoryviews.
        """
        assert isinstance(item, (tuple, slice))

        def verify_slice(s):
            if s.start or s.stop or s.step not in (None, 1):
                raise minierror.InvalidTypeSpecification(
                    "Only a step of 1 may be provided to indicate C or "
                    "Fortran contiguity")

        if isinstance(item, tuple):
            step_idx = None
            for idx, s in enumerate(item):
                verify_slice(s)
                if s.step and (step_idx or idx not in (0, len(item) - 1)):
                    raise minierror.InvalidTypeSpecification(
                        "Step may only be provided once, and only in the "
                        "first or last dimension.")

                if s.step == 1:
                    step_idx = idx

            return ArrayType(self, len(item),
                             is_c_contig=step_idx == len(item) - 1,
                             is_f_contig=step_idx == 0)
        else:
            verify_slice(item)
            return ArrayType(self, 1, is_c_contig=bool(item.step))

    def declare(self):
        return str(self)

    def to_llvm(self, context):
        "Get a corresponding llvm type from this type"
        return context.to_llvm(self)

    def to_ctypes(self):
        import ctypes_conversion
        return ctypes_conversion.convert_to_ctypes(self)

    def get_dtype(self):
        return map_minitype_to_dtype(self)

    def is_string(self):
        return self.is_c_string or self == char.pointer()

    def __getattr__(self, attr):
        if attr.startswith('is_'):
            return False
        return getattr(type(self), attr)

class ArrayType(Type):
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
    subtypes = ['dtype']

    def __init__(self, dtype, ndim, is_c_contig=False, is_f_contig=False,
                 inner_contig=False, broadcasting=None):
        super(ArrayType, self).__init__()
        self.dtype = dtype
        self.ndim = ndim
        self.is_c_contig = is_c_contig
        self.is_f_contig = is_f_contig
        self.inner_contig = inner_contig or is_c_contig or is_f_contig
        self.broadcasting = broadcasting

    @property
    def comparison_type_list(self):
        return [self.dtype, self.is_c_contig, self.is_f_contig, self.inner_contig]

    def pointer(self):
        raise Exception("You probably want a pointer type to the dtype")

    def to_llvm(self, context):
        # raise Exception("Obtain a pointer to the dtype and convert that "
        #                 "to an LLVM type")
        return context.to_llvm(self)

    def __repr__(self):
        axes = [":"] * self.ndim
        if self.is_c_contig:
            axes[-1] = "::1"
        elif self.is_f_contig:
            axes[0] = "::1"

        return "%s[%s]" % (self.dtype, ", ".join(axes))

    @property
    def strided(self):
        type = copy.copy(self)
        type.is_c_contig = False
        type.is_f_contig = False
        type.inner_contig = False
        type.broadcasting = None
        return type

    def __getitem__(self, index):
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


class PointerType(Type):
    is_pointer = True
    subtypes = ['base_type']

    def __init__(self, base_type, **kwds):
        super(PointerType, self).__init__(**kwds)
        self.base_type = base_type

    def __repr__(self):
        return "%s *%s" % (self.base_type, " ".join(self.qualifiers))

    def to_llvm(self, context):
        if self.base_type.is_void:
            llvm_base_type = int_.to_llvm(context)
        else:
            llvm_base_type = self.base_type.to_llvm(context)

        return llvm.core.Type.pointer(llvm_base_type)

class CArrayType(Type):
    is_carray = True
    subtypes = ['base_type']

    def __init__(self, base_type, size, **kwds):
        super(CArrayType, self).__init__(**kwds)
        self.base_type = base_type
        self.size = size

    def __repr__(self):
        return "%s[%d]" % (self.base_type, self.size)

    def to_llvm(self, context):
        return llvm.core.Type.array(self.base_type.to_llvm(context), self.size)

class TypeWrapper(Type):
    is_typewrapper = True
    subtypes = ['opaque_type']

    def __init__(self, opaque_type, context, **kwds):
        super(TypeWrapper, self).__init__(**kwds)
        self.opaque_type = opaque_type
        self.context = context

    def __repr__(self):
        return self.context.declare_type(self)

    def __deepcopy__(self, memo):
        return self

class NamedType(Type):
    name = None

    def __eq__(self, other):
        return isinstance(other, NamedType) and self.name == other.name

    def __repr__(self):
        if self.qualifiers:
            return "%s %s" % (self.name, " ".join(self.qualifiers))
        return str(self.name)

class BoolType(NamedType):
    is_bool = True
    name = "bool"

    def __repr__(self):
        return "int %s" % " ".join(self.qualifiers)

    def to_llvm(self, context):
        return int8.to_llvm(context)

class NumericType(NamedType):
    """
    Base class for numeric types.

    .. attribute:: name

        name of the type

    .. attribute:: itemsize

        sizeof(type)

    .. attribute:: rank

        ordering of numeric types
    """
    is_numeric = True

class IntType(NumericType):
    is_int = True
    is_int_like = True
    name = "int"
    signed = True
    rank = 4
    itemsize = 4

    kind = INT_KIND

    def to_llvm(self, context):
        if self.itemsize == 1:
            return lc.Type.int(8)
        elif self.itemsize == 2:
            return lc.Type.int(16)
        elif self.itemsize == 4:
            return lc.Type.int(32)
        else:
            assert self.itemsize == 8, self
            return lc.Type.int(64)

    def declare(self):
        if self.name.endswith(('16', '32', '64')):
            return self.name + "_t"
        else:
            return str(self)

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

    def to_llvm(self, context):
        if self.itemsize == 4:
            return lc.Type.float()
        elif self.itemsize == 8:
            return lc.Type.double()
        else:
            # Note: what about fp80/fp96?
            assert self.itemsize == 16
            return lc.Type.fp128()

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
        self.itemsize = _plat_bits / 8

class NPyIntp(IntType):
    is_numpy_intp = True
    name = "npy_intp"
    rank = 10

    def __init__(self, **kwds):
        super(NPyIntp, self).__init__(**kwds)
        import numpy as np
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

    def __repr__(self):
        return "PyObject *"

class FunctionType(Type):
    subtypes = ['return_type', 'args']
    is_function = True
    is_vararg = False

    def to_llvm(self, context):
        assert self.return_type is not None
        return lc.Type.function(self.return_type.to_llvm(context),
                                [arg_type.to_llvm(context)
                                    for arg_type in self.args],
                                self.is_vararg)

    def __str__(self):
        args = map(str, self.args)
        if self.is_vararg:
            args.append("...")

        return "%s (*)(%s)" % (self.return_type, ", ".join(args))

class VectorType(Type):
    subtypes = ['element_type']
    is_vector = True
    vector_size = None

    def __init__(self, element_type, vector_size, **kwds):
        super(VectorType, self).__init__(**kwds)
        assert ((element_type.is_int or element_type.is_float) and
                element_type.itemsize in (4, 8)), element_type
        self.element_type = element_type
        self.vector_size = vector_size
        self.itemsize = element_type.itemsize * vector_size

    def to_llvm(self, context):
        return lc.Type.vector(self.element_type.to_llvm(context),
                              self.vector_size)

    @property
    def comparison_type_list(self):
        return self.subtype_list + [self.vector_size]

    def __str__(self):
        itemsize = self.element_type.itemsize
        if self.element_type.is_float:
            if itemsize == 4:
                return '__m128'
            else:
                return '__m128d'
        else:
            if itemsize == 4:
                return '__m128i'
            else:
                raise NotImplementedError

def _sort_key(keyvalue):
    field_name, field_type = keyvalue
    if field_type.is_complex:
        return field_type.base_type.rank * 2
    elif field_type.is_numeric or field_type.is_struct:
        return field_type.rank
    elif field_type.is_vector:
        return _sort_key(field_type.element_type) * field_type.vector_size
    elif field_type.is_array:
        return _sort_key(field_type.base_type) * field_type.size
    elif field_type.is_pointer or field_type.is_object or field_type.is_array:
        return 8
    else:
        return 1

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

class struct(Type):
    """
    Create a struct type. Fields may be ordered or unordered. Unordered fields
    will be ordered from big types to small types (for better alignment).

    >>> struct([('a', int_), ('b', float_)], name='Foo') # ordered struct
    struct Foo { int a, float b }
    >>> struct(a=int_, b=float_, name='Foo') # unordered struct
    struct Foo { float b, int a }
    >>> struct(a=int32, b=int32, name='Foo') # unordered struct
    struct Foo { int32 a, int32 b }

    >>> struct(a=complex128, b=complex64, c=struct(f1=double, f2=double, f3=int32))
    struct { struct { double f1, double f2, int32 f3 } c, complex128 a, complex64 b }
    """

    is_struct = True

    def __init__(self, fields=None, name=None, readonly=False, packed=False, **kwargs):
        super(struct, self).__init__()
        if fields and kwargs:
            raise minierror.InvalidTypeSpecification(
                    "The struct must be either ordered or unordered")

        if kwargs:
            fields = sort_types(kwargs)

        self.fields = fields
        self.rank = sum(_sort_key(field) for field in fields)
        self.name = name
        self.readonly = readonly
        self.fielddict = dict(fields)
        self.packed = packed

    def __repr__(self):
        if self.name:
            name = self.name + ' '
        else:
            name = ''
        return 'struct %s{ %s }' % (
                name, ", ".join("%s %s" % (field_type, field_name)
                                    for field_name, field_type in self.fields))

    def to_llvm(self, context):
        if self.packed:
            lstruct = llvm.core.Type.packed_struct
        else:
            lstruct = llvm.core.Type.struct

        return lstruct([field_type.to_llvm(context)
                           for field_name, field_type in self.fields])

    @property
    def comparison_type_list(self):
        return self.fields

#
### Internal types
#
c_string_type = CStringType()
void = VoidType()

#
### Public types
#
Py_ssize_t = Py_ssize_t_Type()
npy_intp = NPyIntp()
size_t = IntType(name="size_t", rank=8.5, itemsize=8, signed=False)
char = CharType(name="char")
short = IntType(name="short", rank=2, itemsize=struct_.calcsize('h'))
int_ = IntType(name="int", rank=4, itemsize=struct_.calcsize('i'))
long_ = IntType(name="long", rank=5, itemsize=struct_.calcsize('l'))
longlong = IntType(name="PY_LONG_LONG", rank=8, itemsize=struct_.calcsize('q'))

uchar = CharType(name="unsigned char", signed=False)
ushort = IntType(name="unsigned short", rank=2.5,
                 itemsize=struct_.calcsize('H'), signed=False)
uint = IntType(name="unsigned int", rank=4.5,
               itemsize=struct_.calcsize('I'), signed=False)
ulong = IntType(name="unsigned long", rank=5.5,
                itemsize=struct_.calcsize('L'), signed=False)
ulonglong = IntType(name="unsigned PY_LONG_LONG", rank=8.5,
                    itemsize=struct_.calcsize('Q'), signed=False)

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

float32 = float_ = FloatType(name="float", rank=10, itemsize=4)
float64 = double = FloatType(name="double", rank=12, itemsize=8)
float128 = longdouble = FloatType(name="long double", rank=14,
                                  itemsize=16)

complex64 = ComplexType(name="complex64", base_type=float32,
                        rank=16, itemsize=8)
complex128 = ComplexType(name="complex128", base_type=float64,
                         rank=18, itemsize=16)
complex256 = ComplexType(name="complex256", base_type=float128,
                         rank=20, itemsize=32)

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
