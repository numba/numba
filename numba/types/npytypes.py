from __future__ import print_function, division, absolute_import

import numpy

from .abstract import *
from .common import *
from ..typeconv import Conversion


class CharSeq(Type):
    """
    A fixed-length 8-bit character sequence.
    """
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[char x %d]" % count
        super(CharSeq, self).__init__(name)

    @property
    def key(self):
        return self.count


class UnicodeCharSeq(Type):
    """
    A fixed-length unicode character sequence.
    """
    mutable = True

    def __init__(self, count):
        self.count = count
        name = "[unichr x %d]" % count
        super(UnicodeCharSeq, self).__init__(name)

    @property
    def key(self):
        return self.count


class Record(Type):
    """
    A Numpy structured scalar.  *descr* is the string representation
    of the Numpy dtype; *fields* of mapping of field names to
    (type, offset) tuples; *size* the bytesize of a record;
    *aligned* whether the fields are aligned; *dtype* the Numpy dtype
    instance.
    """
    mutable = True

    def __init__(self, descr, fields, size, aligned, dtype):
        self.descr = descr
        self.fields = fields.copy()
        self.size = size
        self.aligned = aligned
        self.dtype = dtype
        name = 'Record(%s)' % descr
        super(Record, self).__init__(name)

    @property
    def key(self):
        # Numpy dtype equality doesn't always succeed, use the descr instead
        # (https://github.com/numpy/numpy/issues/5715)
        return (self.descr, self.size, self.aligned)

    def __len__(self):
        return len(self.fields)

    def offset(self, key):
        return self.fields[key][1]

    def typeof(self, key):
        return self.fields[key][0]

    @property
    def members(self):
        return [(f, t) for f, (t, _) in self.fields.items()]


class DType(DTypeSpec, Opaque):
    """
    Type class for Numpy dtypes.
    """

    def __init__(self, dtype):
        assert isinstance(dtype, Type)
        self._dtype = dtype
        name = "dtype(%s)" % (dtype,)
        super(DTypeSpec, self).__init__(name)

    @property
    def key(self):
        return self.dtype

    @property
    def dtype(self):
        return self._dtype


class NumpyFlatType(SimpleIteratorType, MutableSequence):
    """
    Type class for `ndarray.flat()` objects.
    """

    def __init__(self, arrty):
        self.array_type = arrty
        yield_type = arrty.dtype
        self.dtype = yield_type
        name = "array.flat({arrayty})".format(arrayty=arrty)
        super(NumpyFlatType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.array_type


class NumpyNdEnumerateType(SimpleIteratorType):
    """
    Type class for `np.ndenumerate()` objects.
    """

    def __init__(self, arrty):
        from . import Tuple, UniTuple, intp
        self.array_type = arrty
        yield_type = Tuple((UniTuple(intp, arrty.ndim), arrty.dtype))
        name = "ndenumerate({arrayty})".format(arrayty=arrty)
        super(NumpyNdEnumerateType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.array_type


class NumpyNdIndexType(SimpleIteratorType):
    """
    Type class for `np.ndindex()` objects.
    """

    def __init__(self, ndim):
        from . import UniTuple, intp
        self.ndim = ndim
        yield_type = UniTuple(intp, self.ndim)
        name = "ndindex(dims={ndim})".format(ndim=ndim)
        super(NumpyNdIndexType, self).__init__(name, yield_type)

    @property
    def key(self):
        return self.ndim


class Array(Buffer):
    """
    Type class for Numpy arrays.
    """

    def __init__(self, dtype, ndim, layout, readonly=False, name=None,
                 aligned=True):
        if readonly:
            self.mutable = False
        if (not aligned or
            (isinstance(dtype, Record) and not dtype.aligned)):
            self.aligned = False
        if name is None:
            type_name = "array"
            if not self.mutable:
                type_name = "readonly " + type_name
            if not self.aligned:
                type_name = "unaligned " + type_name
            name = "%s(%s, %sd, %s)" % (type_name, dtype, ndim, layout)
        super(Array, self).__init__(dtype, ndim, layout, name=name)

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if readonly is None:
            readonly = not self.mutable
        return Array(dtype=dtype, ndim=ndim, layout=layout, readonly=readonly,
                     aligned=self.aligned)

    @property
    def key(self):
        return self.dtype, self.ndim, self.layout, self.mutable, self.aligned

    def unify(self, typingctx, other):
        """
        Unify this with the *other* Array.
        """
        if (isinstance(other, Array) and other.ndim == self.ndim
            and other.dtype == self.dtype):
            if self.layout == other.layout:
                layout = self.layout
            else:
                layout = 'A'
            readonly = not (self.mutable and other.mutable)
            aligned = self.aligned and other.aligned
            return Array(dtype=self.dtype, ndim=self.ndim, layout=layout,
                         readonly=readonly, aligned=aligned)

    def can_convert_to(self, typingctx, other):
        """
        Convert this Array to the *other*.
        """
        if (isinstance(other, Array) and other.ndim == self.ndim
            and other.dtype == self.dtype):
            if (other.layout in ('A', self.layout)
                and (self.mutable or not other.mutable)
                and (self.aligned or not other.aligned)):
                return Conversion.safe

class SmartArrayType(Array):

    def __init__(self, dtype, ndim, layout, pyclass):
        self.pyclass = pyclass
        super(SmartArrayType, self).__init__(dtype, ndim, layout, name='numba_array')

    @property
    def as_array(self):
        return Array(self.dtype, self.ndim, self.layout)

    def copy(self, dtype=None, ndim=None, layout=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        return type(self)(dtype, ndim, layout, self.pyclass)

class ArrayCTypes(Type):
    """
    This is the type for `numpy.ndarray.ctypes`.
    """
    def __init__(self, arytype):
        # This depends on the ndim for the shape and strides attributes,
        # even though they are not implemented, yet.
        self.ndim = arytype.ndim
        name = "ArrayCTypes(ndim={0})".format(self.ndim)
        super(ArrayCTypes, self).__init__(name)

    @property
    def key(self):
        return self.ndim


class ArrayFlags(Type):
    """
    This is the type for `numpy.ndarray.flags`.
    """
    def __init__(self, arytype):
        self.array_type = arytype
        name = "ArrayFlags({0})".format(self.array_type)
        super(ArrayFlags, self).__init__(name)

    @property
    def key(self):
        return self.array_type


class NestedArray(Array):
    """
    A NestedArray is an array nested within a structured type (which are "void"
    type in NumPy parlance). Unlike an Array, the shape, and not just the number
    of dimenions is part of the type of a NestedArray.
    """

    def __init__(self, dtype, shape):
        assert dtype.bitwidth % 8 == 0, \
            "Dtype bitwidth must be a multiple of bytes"
        self._shape = shape
        name = "nestedarray(%s, %s)" % (dtype, shape)
        ndim = len(shape)
        super(NestedArray, self).__init__(dtype, ndim, 'C', name=name)

    @property
    def shape(self):
        return self._shape

    @property
    def nitems(self):
        l = 1
        for s in self.shape:
            l = l * s
        return l

    @property
    def size(self):
        return self.dtype.bitwidth // 8

    @property
    def strides(self):
        stride = self.size
        strides = []
        for i in reversed(self._shape):
             strides.append(stride)
             stride *= i
        return tuple(reversed(strides))

    @property
    def key(self):
        return self.dtype, self.shape
