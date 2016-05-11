from __future__ import print_function, division, absolute_import

import collections

import numpy as np

from .abstract import *
from .common import *
from ..typeconv import Conversion
from .. import utils


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


class NumpyNdIterType(IteratorType):
    """
    Type class for `np.nditer()` objects.

    The layout denotes in which order the logical shape is iterated on.
    "C" means logical order (corresponding to in-memory order in C arrays),
    "F" means reverse logical order (corresponding to in-memory order in
    F arrays).
    """

    def __init__(self, arrays):
        # Note inputs arrays can also be scalars, in which case they are
        # broadcast.
        self.arrays = tuple(arrays)
        self.layout = self._compute_layout(self.arrays)
        self.dtypes = tuple(getattr(a, 'dtype', a) for a in self.arrays)
        self.ndim = max(getattr(a, 'ndim', 0) for a in self.arrays)
        name = "nditer(ndim={ndim}, layout={layout}, inputs={arrays})".format(
            ndim=self.ndim, layout=self.layout, arrays=self.arrays)
        super(NumpyNdIterType, self).__init__(name)

    @classmethod
    def _compute_layout(cls, arrays):
        c = collections.Counter()
        for a in arrays:
            if not isinstance(a, Array):
                continue
            if a.layout in 'CF' and a.ndim == 1:
                c['C'] += 1
                c['F'] += 1
            elif a.ndim >= 1:
                c[a.layout] += 1
        return 'F' if c['F'] > c['C'] else 'C'

    @property
    def key(self):
        return self.arrays

    @property
    def views(self):
        """
        The views yielded by the iterator.
        """
        return [Array(dtype, 0, 'C') for dtype in self.dtypes]

    @property
    def yield_type(self):
        from . import BaseTuple
        views = self.views
        if len(views) > 1:
            return BaseTuple.from_types(views)
        else:
            return views[0]

    @utils.cached_property
    def indexers(self):
        """
        A list of (kind, start_dim, end_dim, indices) where:
        - `kind` is either "flat", "indexed", "0d" or "scalar"
        - `start_dim` and `end_dim` are the dimension numbers at which
          this indexing takes place
        - `indices` is the indices of the indexed arrays in self.arrays
        """
        d = collections.OrderedDict()
        layout = self.layout
        ndim = self.ndim
        assert layout in 'CF'
        for i, a in enumerate(self.arrays):
            if not isinstance(a, Array):
                indexer = ('scalar', 0, 0)
            elif a.ndim == 0:
                indexer = ('0d', 0, 0)
            else:
                if a.layout == layout or (a.ndim == 1 and a.layout in 'CF'):
                    kind = 'flat'
                else:
                    kind = 'indexed'
                if layout == 'C':
                    # If iterating in C order, broadcasting is done on the outer indices
                    indexer = (kind, ndim - a.ndim, ndim)
                else:
                    indexer = (kind, 0, a.ndim)
            d.setdefault(indexer, []).append(i)
        return list(k + (v,) for k, v in d.items())

    @utils.cached_property
    def need_shaped_indexing(self):
        """
        Whether iterating on this iterator requires keeping track of
        individual indices inside the shape.  If False, only a single index
        over the equivalent flat shape is required, which can make the
        iterator more efficient.
        """
        for kind, start_dim, end_dim, _ in self.indexers:
            if kind in ('0d', 'scalar'):
                pass
            elif kind == 'flat':
                if (start_dim, end_dim) != (0, self.ndim):
                    # Broadcast flat iteration needs shaped indexing
                    # to know when to restart iteration.
                    return True
            else:
                return True
        return False


class NumpyNdIndexType(SimpleIteratorType):
    """
    Type class for `np.ndindex()` objects.
    """

    def __init__(self, ndim):
        from . import UniTuple, intp
        self.ndim = ndim
        yield_type = UniTuple(intp, self.ndim)
        name = "ndindex(ndim={ndim})".format(ndim=ndim)
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
    This is the type for `np.ndarray.ctypes`.
    """
    def __init__(self, arytype):
        # This depends on the ndim for the shape and strides attributes,
        # even though they are not implemented, yet.
        self.dtype = arytype.dtype
        self.ndim = arytype.ndim
        name = "ArrayCTypes(dtype={0}, ndim={1})".format(self.dtype, self.ndim)
        super(ArrayCTypes, self).__init__(name)

    @property
    def key(self):
        return self.dtype, self.ndim

    def can_convert_to(self, typingctx, other):
        """
        Convert this type to the corresponding pointer type.
        This allows passing a array.ctypes object to a C function taking
        a raw pointer.

        Note that in pure Python, the array.ctypes object can only be
        passed to a ctypes function accepting a c_void_p, not a typed
        pointer.
        """
        from . import CPointer, voidptr
        # XXX what about readonly
        if isinstance(other, CPointer) and other.dtype == self.dtype:
            return Conversion.safe
        elif other == voidptr:
            return Conversion.safe


class ArrayFlags(Type):
    """
    This is the type for `np.ndarray.flags`.
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
