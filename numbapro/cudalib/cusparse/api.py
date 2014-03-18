from __future__ import print_function, absolute_import, division
from contextlib import contextmanager
import numpy as np
from .binding import (cuSparse, CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_INDEX_BASE_ONE)
from numbapro import cuda

dtype_to_char = {
    np.dtype(np.float32): 'S',
    np.dtype(np.float64): 'D',
    np.dtype(np.complex64): 'C',
    np.dtype(np.complex128): 'Z',
}


def _sentry_ndim(ndim, **kws):
    for k, a in kws.items():
        if a.ndim != ndim:
            raise ValueError("%s.ndim must be %dD" % (k, ndim))


def _sentry_dtype(dtype, **kws):
    for k, a in kws.items():
        if a.dtype != dtype:
            raise TypeError("%s.dtype is not %s" % (k, dtype))


@contextmanager
def _readonly(*arys):
    ds = []
    for a in arys:
        dmem, _ = cuda._auto_device(a)
        ds.append(dmem)
    yield ds


@contextmanager
def _readwrite(*arys):
    ds = []
    ws = []
    for a in arys:
        dmem, conv = cuda._auto_device(a)
        ds.append(dmem)
        if conv:
            ws.append((a, dmem))
    yield ds
    for a, d in ws:
        d.copy_to_host(a)


class Sparse(object):
    @cuda.require_context
    def __init__(self, idxbase=0):
        """
        Args
        ----
        - idxbase int
            Index base. Must be 0 or 1
        """
        if idxbase not in (0, 1):
            raise ValueError("Invalid index base")

        self.api = cuSparse()
        self.idxbase = (CUSPARSE_INDEX_BASE_ZERO,
                        CUSPARSE_INDEX_BASE_ONE)[idxbase]

    def _get_api(self, fname, dtype):
        ch = dtype_to_char[np.dtype(dtype)]
        fn = "%s%s" % (ch, fname)
        return getattr(self.api, fn)

    def axpyi(self, alpha, xval, xind, y):
        _sentry_ndim(1, xval=xval, xind=xval, y=y)
        _sentry_dtype(np.int32, xind=xind)
        _sentry_dtype(xval.dtype, y=y)
        fn = self._get_api("axpyi", xval.dtype)
        nnz = xval.size
        with _readonly(xval, xind) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                fn(nnz=nnz, alpha=alpha, xVal=dxval, xInd=dxind, y=dy,
                   idxBase=self.idxbase)
        return y

    def doti(self, xval, xind, y):
        _sentry_ndim(1, xval=xval, xind=xind, y=y)
        _sentry_dtype(np.int32, xind=xind)
        _sentry_dtype(xval.dtype, y=y)
        fn = self._get_api("doti", xval.dtype)
        nnz = xval.size
        with _readonly(xval, xind) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                result = fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy,
                            idxBase=self.idxbase)
        return result

    def dotci(self, xval, xind, y):
        _sentry_ndim(1, xval=xval, xind=xind, y=y)
        _sentry_dtype(np.int32, xind=xind)
        _sentry_dtype(xval.dtype, y=y)
        fn = self._get_api("dotci", xval.dtype)
        nnz = xval.size
        with _readonly(xval, xind) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                result = fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy,
                            idxBase=self.idxbase)
        return result

    def gthr(self, y, xval, xind):
        _sentry_ndim(1, xval=xval, xind=xind, y=y)
        _sentry_dtype(np.int32, xind=xind)
        _sentry_dtype(xval.dtype, y=y)
        fn = self._get_api("gthr", xval.dtype)
        nnz = xval.size
        with _readonly(y, xind) as [dy, dxind]:
            with _readwrite(xval) as [dxval]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, idxBase=self.idxbase)

    def gthrz(self, y, xval, xind):
        _sentry_ndim(1, xval=xval, xind=xind, y=y)
        _sentry_dtype(np.int32, xind=xind)
        _sentry_dtype(xval.dtype, y=y)
        fn = self._get_api("gthrz", xval.dtype)
        nnz = xval.size
        with _readonly(xind) as [dxind]:
            with _readwrite(y, xval) as [dy, dxval]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, idxBase=self.idxbase)

    def roti(self, xval, xind, y, c, s):
        _sentry_ndim(1, xval=xval, xind=xind, y=y)
        _sentry_dtype(np.int32, xind=xind)
        _sentry_dtype(xval.dtype, y=y)
        fn = self._get_api("roti", xval.dtype)
        nnz = xval.size
        with _readonly(xind) as [dxind]:
            with _readwrite(y, xval) as [dy, dxval]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, c=c, s=s,
                   idxBase=self.idxbase)

    def sctr(self, xval, xind, y):
        _sentry_ndim(1, xval=xval, xind=xind, y=y)
        _sentry_dtype(np.int32, xind=xind)
        _sentry_dtype(xval.dtype, y=y)
        fn = self._get_api("sctr", xval.dtype)
        nnz = xval.size
        with _readonly(xval, xind) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, idxBase=self.idxbase)


def coo_matrix(ary):
    """
    Args
    -----
    - ary [np.ndarray]
        Host array

    """
