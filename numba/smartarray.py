from numba.tracing import trace
from numba.errors import deprecated

import sys

import numpy as np

def _o2s(dtype, shape, order):
    # convert order parameter to strides

    if dtype is None or shape is None or order is None:
        return None

    if order == 'F':
        shape = list(shape)
        shape.reverse()
    strides = []
    itemsize = dtype.itemsize
    for i in range(len(shape), 0, -1):
        strides.append(itemsize)
        itemsize *= shape[i - 1]
    if order in ('C', None):
        strides.reverse()
    return tuple(strides)

def _s2o(dtype, shape, strides):
    # convert strides parameter to order
    # Note: strides must correspond to contiguous data layout

    if strides is None or strides[-1] == dtype.itemsize:
        order = 'C'
    elif strides[0] == dtype.itemsize:
        order = 'F'
    else:
        raise ValueError('strides do not correspond to contiguous data layout')
    s2 = _o2s(dtype, shape, order)
    if strides != s2:
        raise ValueError('strides do not correspond to contiguous data layout')
    return order

class SmartArray(object):
    """An array type that supports host and GPU storage."""

    _targets = ('host', 'gpu')

    def __init__(self, obj=None, copy=True,
                 shape=None, dtype=None, order=None, where='host'):
        """Construct a SmartArray in the memory space defined by 'where'.
        Valid invocations:

        * SmartArray(obj=<array-like object>, copy=<optional-true-or-false>):

          to create a SmartArray from an existing array-like object.
          The 'copy' argument specifies whether to adopt or to copy it.

        * SmartArray(shape=<shape>, dtype=<dtype>, order=<order>)

          to create a new SmartArray from scratch, given the typical NumPy array
          attributes.

        (The optional 'where' argument specifies where to allocate the array
        initially. (Default: 'host')
        """

        if where not in self._targets:
            raise ValueError('"%s" is not a valid target'%where)
        # we need either a prototype or proper type info
        assert obj is not None or (shape and dtype)
        self._host = self._gpu = None
        self._host_valid = self._gpu_valid = False
        self._allocate(where, obj, dtype, shape, _o2s(dtype, shape, order), copy)
        if where == 'host':
            self._host_valid = True
            t = self._host
        else:
            self._gpu_valid = True
            t = self._gpu
        self._shape = t.shape
        self._strides = t.strides
        self._dtype = t.dtype
        self._ndim = t.ndim
        self._size = t.size

    @property
    def shape(self): return self._shape

    @property
    def strides(self): return self._strides

    @property
    def dtype(self): return self._dtype

    @property
    def ndim(self): return self._ndim

    @property
    def size(self): return self._size

    def get(self, where='host'):
        """Return the representation of 'self' in the given memory space."""

        if where not in self._targets:
            raise ValueError('"%s" is not a valid target'%where)
        self._sync(where)
        if where == 'host': return self._host
        elif where == 'gpu': return self._gpu
        else: raise ValueError('unknown memory space "%s"'%where)

    @deprecated("get('host')")
    def host(self): return self.get('host')
    @deprecated("get('gpu')")
    def gpu(self): return self.get('gpu')

    def mark_changed(self, where='host'):
        """Mark the given location as changed, broadcast updates if needed."""

        if where not in self._targets:
            raise ValueError('"%s" is not a valid target'%where)
        if where == 'host':
            self._invalidate('gpu')
            # only sync if there are active views
            if self._gpu is not None and sys.getrefcount(self._gpu) > 2:
                self._sync('gpu')
        elif where == 'gpu':
            self._invalidate('host')
            # only sync if there are active views
            if self._host is not None and sys.getrefcount(self._host) > 2:
                self._sync('host')

    @deprecated("mark_changed('host')")
    def host_changed(self): return self.mark_changed('host')
    @deprecated("mark_changed('gpu')")
    def gpu_changed(self): return self.mark_changed('gpu')

    def __array__(self, *args):

        self._sync('host')
        return np.array(self._host, *args)

    def _sync(self, where):
        """Sync the data in one memory space with the other."""

        if where == 'gpu':
            if self._gpu is None:
                self._allocate('gpu', None, self.dtype, self.shape, self.strides)
            if not self._gpu_valid:
                self._copy_to_gpu()
        else:
            if self._host is None:
                self._allocate('host', None, self.dtype, self.shape, self.strides)
            if not self._host_valid:
                self._copy_to_host()

    @trace
    def _invalidate(self, where):
        """Mark the host / device array as out-of-date."""

        if where == 'gpu':
            self._gpu_valid = False
        else:
            self._host_valid = False

    @trace
    def _allocate(self, where, obj=None, dtype=None, shape=None, strides=None,
                  copy=True):
        if dtype:
            dtype = np.dtype(dtype)
        if where == 'host':
            if obj is not None:
                self._host = np.array(obj, dtype, copy=copy)
            else:
                self._host = np.empty(shape, dtype, _s2o(dtype, shape, strides))
        else:
            # Don't import this at module-scope as it may not be available
            # in all environments (e.g., CUDASIM)
            from numba.cuda.cudadrv import devicearray as da
            if obj is not None:
                # If 'obj' is an array-like object but not an ndarray,
                # construct an ndarray first to extract all the parameters we need.
                if not isinstance(obj, np.ndarray):
                    obj = np.array(obj, copy=False)
                self._gpu = da.from_array_like(obj)
            else:
                if strides is None:
                    strides = _o2s(dtype, shape, 'C')
                self._gpu = da.DeviceNDArray(shape, strides, dtype)

    @trace
    def _copy_to_gpu(self):
        self._gpu.copy_to_device(self._host)
        self._gpu_valid = True

    @trace
    def _copy_to_host(self):
        self._gpu.copy_to_host(self._host)
        self._host_valid = True

    @staticmethod
    def _maybe_wrap(value):
        """If `value` is an ndarray, wrap it in a SmartArray,
        otherwise return `value` itself."""

        if isinstance(value, np.ndarray):
            return SmartArray(value, copy=False)
        else:
            return value
        
    @trace
    def __getattr__(self, name):
        """Transparently forward attribute access to the host array."""

        if self._host is None:
            self._allocate('host', None, self.dtype, self.shape, self.strides)

        # FIXME: for some attributes we need to sync first !
        return self._maybe_wrap(getattr(self._host, name))

    def __len__(self): return self.shape[0]
    def __eq__(self, other):
        if type(self) is not type(other): return False
        # FIXME: If both arrays have valid GPU data, compare there.
        return self._maybe_wrap(self.get('host') == other.get('host'))
    def __getitem__(self, *args):
        return self._maybe_wrap(self.get('host').__getitem__(*args))
    def __setitem__(self, *args):
        return self._maybe_wrap(self.get('host').__setitem__(*args))
    def astype(self, *args):
        return self._maybe_wrap(self.get('host').astype(*args))
