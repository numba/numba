from __future__ import print_function, division

from collections import namedtuple
import itertools
import functools
import operator

import numpy as np


Extent = namedtuple("Extent", ["begin", "end"])


class Dim(object):
    """A single dimension of the array

    Attributes
    ----------
    start:
        start offset
    stop:
        stop offset
    size:
        number of items
    stride:
        item stride
    """
    __slots__ = 'start', 'stop', 'size', 'stride', 'single'

    def __init__(self, start, stop, size, stride, single):
        if stop < start:
            raise ValueError("end offset is before start offset")
        self.start = start
        self.stop = stop
        self.size = size
        self.stride = stride
        self.single = single
        assert not single or size == 1

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            single = False
        else:
            single = True
            start = item
            stop = start + 1
            step = None

        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        if step is None:
            step = 1

        stride = step * self.stride

        if start >= 0:
            start = self.start + start * self.stride
        else:
            start = self.stop + start * self.stride

        if stop >= 0:
            stop = self.start + stop * self.stride
        else:
            stop = self.stop + stop * self.stride

        size = (stop - start) // stride

        if self.start >= start >= self.stop:
            raise IndexError("start index out-of-bound")

        if self.start >= stop >= self.stop:
            raise IndexError("stop index out-of-bound")

        if stop < start:
            start = stop
            size = 0

        return Dim(start, stop, size, stride, single)

    def get_offset(self, idx):
        return self.start + idx * self.stride

    def __repr__(self):
        strfmt = "Dim(start=%s, stop=%s, size=%s, stride=%s)"
        return strfmt % (self.start, self.stop, self.size, self.stride)

    def normalize(self, base):
        return Dim(start=self.start - base, stop=self.stop - base,
                   size=self.size, stride=self.stride, single=self.single)

    def copy(self, start=None, stop=None, size=None, stride=None, single=None):
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        if size is None:
            size = self.size
        if stride is None:
            stride = self.stride
        if single is None:
            single = self.single
        return Dim(start, stop, size, stride, single)

    def is_contiguous(self, itemsize):
        return self.stride == itemsize


def compute_index(indices, dims):
    return sum(d.get_offset(i) for i, d in zip(indices, dims))


class Element(object):
    is_array = False

    def __init__(self, extent):
        self.extent = extent

    def iter_contiguous_extent(self):
        yield self.extent


class Array(object):
    """A dummy numpy array-like object.  Consider it an array without the
    actual data, but offset from the base data pointer.

    Attributes
    ----------
    dims: tuple of Dim
        describing each dimension of the array

    ndim: int
        number of dimension

    shape: tuple of int
        size of each dimension

    strides: tuple of int
        stride of each dimension

    itemsize: int
        itemsize

    extent: (start, end)
        start and end offset containing the memory region
    """
    is_array = True

    @classmethod
    def from_desc(cls, offset, shape, strides, itemsize):
        dims = []
        for ashape, astride in zip(shape, strides):
            dim = Dim(offset, offset + ashape * astride, ashape, astride,
                      single=False)
            dims.append(dim)
        return cls(dims, itemsize)

    def __init__(self, dims, itemsize):
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.shape = tuple(dim.size for dim in self.dims)
        self.strides = tuple(dim.stride for dim in self.dims)
        self.itemsize = itemsize
        self.size = np.prod(self.shape)
        self.extent = self._compute_extent()
        self.flags = self._compute_layout()

    def _compute_layout(self):
        flags = {}

        if not self.dims:
            # Records have no dims, and we can treat them as contiguous
            flags['F_CONTIGUOUS'] = True
            flags['C_CONTIGUOUS'] = True
            return flags

        leftmost = self.dims[0].is_contiguous(self.itemsize)
        rightmost = self.dims[-1].is_contiguous(self.itemsize)

        def is_contig(traverse):
            last = next(traverse)
            for dim in traverse:
                if last.size != 0 and last.size * last.stride != dim.stride:
                    return False
                last = dim
            return True

        flags['F_CONTIGUOUS'] = leftmost and is_contig(iter(self.dims))
        flags['C_CONTIGUOUS'] = rightmost and is_contig(reversed(self.dims))
        return flags

    def _compute_extent(self):
        firstidx = [0] * self.ndim
        lastidx = [s - 1 for s in self.shape]
        start = compute_index(firstidx, self.dims)
        stop = compute_index(lastidx, self.dims) + self.itemsize
        return Extent(start, stop)

    def __repr__(self):
        return '<Array dims=%s itemsize=%s>' % (self.dims, self.itemsize)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = [item]
        else:
            item = list(item)

        nitem = len(item)
        ndim = len(self.dims)
        if nitem > ndim:
            raise IndexError("%d extra indices given" % (nitem - ndim,))

        # Add empty slices for missing indices
        while len(item) < ndim:
            item.append(slice(None, None))

        dims = [dim.__getitem__(it) for dim, it in zip(self.dims, item)]
        newshape = [d.size for d in dims if not d.single]
        arr = Array(dims, self.itemsize)
        if newshape:
            return arr.reshape(*newshape)[0]
        else:
            return Element(arr.extent)

    @property
    def is_c_contig(self):
        return self.flags['C_CONTIGUOUS']

    @property
    def is_f_contig(self):
        return self.flags['F_CONTIGUOUS']

    def iter_contiguous_extent(self):
        """ Generates extents
        """
        if self.is_c_contig or self.is_f_contig:
            yield self.extent
        else:
            if self.dims[0].stride < self.dims[-1].stride:
                innerdim = self.dims[0]
                outerdims = self.dims[1:]
                outershape = self.shape[1:]
            else:
                innerdim = self.dims[-1]
                outerdims = self.dims[:-1]
                outershape = self.shape[:-1]

            if innerdim.is_contiguous(self.itemsize):
                oslen = [range(s) for s in outershape]
                for indices in itertools.product(*oslen):
                    base = compute_index(indices, outerdims)
                    yield base + innerdim.start, base + innerdim.stop
            else:
                oslen = [range(s) for s in self.shape]
                for indices in itertools.product(*oslen):
                    offset = compute_index(indices, self.dims)
                    yield offset, offset + self.itemsize

    def reshape(self, *newshape, **kws):
        oldnd = self.ndim
        newnd = len(newshape)

        if newshape == self.shape:
            return self, None

        order = kws.pop('order', 'C')
        if kws:
            raise TypeError('unknown keyword arguments %s' % kws.keys())
        if order not in 'CFA':
            raise ValueError('order not C|F|A')

        newsize = functools.reduce(operator.mul, newshape, 1)

        if order == 'A':
            order = 'F' if self.is_f_contig else 'C'

        if newsize != self.size:
            raise ValueError("reshape changes the size of the array")

        elif newnd == 1 or self.is_c_contig or self.is_f_contig:
            if order == 'C':
                newstrides = list(iter_strides_c_contig(self, newshape))
            elif order == 'F':
                newstrides = list(iter_strides_f_contig(self, newshape))
            else:
                raise AssertionError("unreachable")

            ret = self.from_desc(self.extent.begin, shape=newshape,
                                 strides=newstrides, itemsize=self.itemsize)

            return ret, list(self.iter_contiguous_extent())
        else:
            raise NotImplementedError("reshape on non-contiguous array")

    def ravel(self, order='C'):
        if order not in 'CFA':
            raise ValueError('order not C|F|A')

        if self.ndim <= 1:
            return self

        elif (order == 'C' and self.is_c_contig or
                          order == 'F' and self.is_f_contig):
            newshape = (self.size,)
            newstrides = (self.itemsize,)
            arr = self.from_desc(self.extent.begin, newshape, newstrides,
                                 self.itemsize)
            return arr, list(self.iter_contiguous_extent())

        else:
            raise NotImplementedError("ravel on non-contiguous array")


def iter_strides_f_contig(arr, shape=None):
    """yields the f-contigous strides
    """
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize
    yield itemsize
    sum = 1
    for s in shape[:-1]:
        sum *= s
        yield sum * itemsize


def iter_strides_c_contig(arr, shape=None):
    """yields the c-contigous strides
    """
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize

    def gen():
        yield itemsize
        sum = 1
        for s in reversed(shape[1:]):
            sum *= s
            yield sum * itemsize

    for i in reversed(list(gen())):
        yield i


def is_element_indexing(item, ndim):
    if isinstance(item, slice):
        return False

    elif isinstance(item, tuple):
        if len(item) == ndim:
            if not any(isinstance(it, slice) for it in item):
                return True

    else:
        return True

    return False

