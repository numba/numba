from __future__ import print_function, division
import itertools
import numpy


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
    __slots__ = 'start', 'stop', 'size', 'stride'

    def __init__(self, start, stop, size, stride):
        if stop < start:
            raise ValueError("end offset is before start offset")
        self.start = start
        self.stop = stop
        self.size = size
        self.stride = stride

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
        else:
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

        return Dim(start, stop, size, stride)

    def get_offset(self, idx):
        return self.start + idx * self.stride

    def __repr__(self):
        strfmt = "Dim(start=%s, stop=%s, size=%s, stride=%s)"
        return strfmt % (self.start, self.stop, self.size, self.stride)

    def normalize(self, base):
        return Dim(start=self.start - base, stop=self.stop - base,
                   size=self.size, stride=self.stride)

    def copy(self, start=None, stop=None, size=None, stride=None):
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        if size is None:
            size = self.size
        if stride is None:
            stride = self.stride
        return Dim(start, stop, size, stride)

    def is_contiguous(self, itemsize):
        return self.stride == itemsize


def compute_index(indices, dims):
    return sum(d.get_offset(i) for i, d in zip(indices, dims))


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

    @classmethod
    def from_desc(cls, offset, shape, strides, itemsize):
        dims = []
        for ashape, astride in zip(shape, strides):
            dim = Dim(offset, offset + ashape * astride, ashape, astride)
            dims.append(dim)
        return cls(dims, itemsize)

    def __init__(self, dims, itemsize):
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.shape = tuple(dim.size for dim in self.dims)
        self.strides = tuple(dim.stride for dim in self.dims)
        self.itemsize = itemsize
        self.size = numpy.prod(self.shape)
        self.extent = self._compute_extent()
        self.flags = self._compute_layout()

    def _compute_layout(self):
        leftmost = self.dims[0].is_contiguous(self.itemsize)
        rightmost = self.dims[-1].is_contiguous(self.itemsize)
        flags = {}

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
        return start, stop

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
        return Array(dims, self.itemsize)

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
                oslen = [xrange(s) for s in outershape]
                for indices in itertools.product(*oslen):
                    base = compute_index(indices, outerdims)
                    yield base + innerdim.start, base + innerdim.stop
            else:
                oslen = [xrange(s) for s in self.shape]
                for indices in itertools.product(*oslen):
                    offset = compute_index(indices, self.dims)
                    yield offset, offset + self.itemsize
