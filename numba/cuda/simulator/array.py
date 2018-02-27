from contextlib import contextmanager
import numpy as np
from warnings import warn
from numba.six import raise_from


errmsg_contiguous_buffer = ("Array contains non-contiguous buffer and cannot "
                            "be transferred as a single memory region. Please "
                            "ensure contiguous buffer with numpy "
                            ".ascontiguousarray()")

class FakeShape(tuple):
    '''
    The FakeShape class is used to provide a shape which does not allow negative
    indexing, similar to the shape in CUDA Python. (Numpy shape arrays allow
    negative indexing)
    '''
    def __getitem__(self, k):
        if k < 0:
            raise IndexError('tuple index out of range')
        return super(FakeShape, self).__getitem__(k)


class FakeCUDAArray(object):
    '''
    Implements the interface of a DeviceArray/DeviceRecord, but mostly just
    wraps a NumPy array.
    '''

    __cuda_ndarray__ = True     # There must be gpu_data attribute


    def __init__(self, ary, stream=0):
        self._ary = ary
        self.stream = stream

    def __getattr__(self, attrname):
        try:
            attr = getattr(self._ary, attrname)
            return attr
        except AttributeError as e:
            raise_from(AttributeError("Wrapped array has no attribute '%s'"
                                      % attrname), e)

    def bind(self, stream=0):
        return FakeCUDAArray(self._ary, stream)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        return FakeCUDAArray(np.transpose(self._ary, axes=axes))

    def __getitem__(self, idx):
        item = self._ary.__getitem__(idx)
        if isinstance(item, np.ndarray):
            return FakeCUDAArray(item, stream=self.stream)
        return item

    def __setitem__(self, idx, val):
        return self._ary.__setitem__(idx, val)

    def copy_to_host(self, ary=None, stream=0):
        if ary is None:
            ary = np.empty_like(self._ary)
        # NOTE: np.copyto() introduced in Numpy 1.7
        try:
            np.copyto(ary, self._ary)
        except AttributeError:
            ary[:] = self._ary
        return ary

    def copy_to_device(self, ary, stream=0):
        '''
        Copy from the provided array into this array.

        This may be less forgiving than the CUDA Python implementation, which
        will copy data up to the length of the smallest of the two arrays,
        whereas this expects the size of the arrays to be equal.
        '''
        sentry_contiguous(self)
        if isinstance(ary, FakeCUDAArray):
            sentry_contiguous(ary)

            if self.flags['C_CONTIGUOUS'] != ary.flags['C_CONTIGUOUS']:
                raise ValueError("Can't copy %s-contiguous array to a %s-contiguous array" % (
                    'C' if ary.flags['C_CONTIGUOUS'] else 'F',
                    'C' if self.flags['C_CONTIGUOUS'] else 'F',
                ))
        else:
            ary = np.array(
                ary,
                order='C' if self.flags['C_CONTIGUOUS'] else 'F',
                subok=True,
                copy=False)
        try:
            np.copyto(self._ary, ary)
        except AttributeError:
            self._ary[:] = ary

    def to_host(self):
        warn('to_host() is deprecated and will be removed')

    @property
    def shape(self):
        return FakeShape(self._ary.shape)

    def ravel(self, *args, **kwargs):
        return FakeCUDAArray(self._ary.ravel(*args, **kwargs))

    def reshape(self, *args, **kwargs):
        return FakeCUDAArray(self._ary.reshape(*args, **kwargs))

    def is_c_contiguous(self):
        return self._ary.flags.c_contiguous

    def is_f_contiguous(self):
        return self._ary.flags.f_contiguous

    def __str__(self):
        return str(self._ary)

    def __repr__(self):
        return repr(self._ary)

    def __len__(self):
        return len(self._ary)

    def split(self, section, stream=0):
        return [
            FakeCUDAArray(a)
            for a in np.split(self._ary, range(section, len(self), section))
        ]

def sentry_contiguous(ary):
    if not ary.flags['C_CONTIGUOUS'] and not ary.flags['F_CONTIGUOUS']:
        if ary.strides[0] == 0:
            # Broadcasted, ensure inner contiguous
            return sentry_contiguous(ary[0])
        else:
            raise ValueError(errmsg_contiguous_buffer)


def to_device(ary, stream=0, copy=True, to=None):
    sentry_contiguous(ary)
    if to is None:
        return FakeCUDAArray(np.array(
            ary,
            order='C' if ary.flags['C_CONTIGUOUS'] else 'F', copy=True))
    else:
        to.copy_to_device(ary, stream=stream)


@contextmanager
def pinned(arg):
    yield


def pinned_array(shape, dtype=np.float, strides=None, order='C'):
    return np.ndarray(shape=shape, strides=strides, dtype=dtype, order=order)


def device_array(*args, **kwargs):
    stream = kwargs.pop('stream') if 'stream' in kwargs else 0
    return FakeCUDAArray(np.ndarray(*args, **kwargs), stream=stream)


def device_array_like(ary, stream=0):
    return FakeCUDAArray(np.empty_like(ary))


# Fake devicearray.auto_device
class devicearray(object):
    @staticmethod
    def auto_device(ary, stream=0, copy=True):
        if not isinstance(ary, np.void):
            ary = np.array(
                ary,
                copy=False,
                subok=True)
        return to_device(ary, stream, copy), False


