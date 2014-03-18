from __future__ import print_function, absolute_import, division
from contextlib import contextmanager
import numpy as np
import scipy.sparse as ss
from numbapro import cuda
from .binding import (cuSparse, CUSPARSE_INDEX_BASE_ZERO,
                      CUSPARSE_INDEX_BASE_ONE)

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

    def matdescr(self, indexbase=None, diagtype='N', fillmode='L',
                 matrixtype='G'):
        descr = self.api.matdescr()
        descr.indexbase = self.idxbase if indexbase is None else indexbase
        descr.diagtype = diagtype
        descr.fillmode = fillmode
        descr.matrixtype = matrixtype
        return descr

    # ------------------------------------------------------------------------
    # Level 1 API

    def axpyi(self, alpha, xVal, xInd, y):
        _sentry_ndim(1, xVal=xVal, xInd=xVal, y=y)
        _sentry_dtype(np.int32, xInd=xInd)
        _sentry_dtype(xVal.dtype, y=y)
        fn = self._get_api("axpyi", xVal.dtype)
        nnz = xVal.size
        with _readonly(xVal, xInd) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                fn(nnz=nnz, alpha=alpha, xVal=dxval, xInd=dxind, y=dy,
                   idxBase=self.idxbase)
        return y

    def doti(self, xVal, xInd, y):
        _sentry_ndim(1, xVal=xVal, xInd=xInd, y=y)
        _sentry_dtype(np.int32, xInd=xInd)
        _sentry_dtype(xVal.dtype, y=y)
        fn = self._get_api("doti", xVal.dtype)
        nnz = xVal.size
        with _readonly(xVal, xInd) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                result = fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy,
                            idxBase=self.idxbase)
        return result

    def dotci(self, xVal, xInd, y):
        _sentry_ndim(1, xVal=xVal, xInd=xInd, y=y)
        _sentry_dtype(np.int32, xInd=xInd)
        _sentry_dtype(xVal.dtype, y=y)
        fn = self._get_api("dotci", xVal.dtype)
        nnz = xVal.size
        with _readonly(xVal, xInd) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                result = fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy,
                            idxBase=self.idxbase)
        return result

    def gthr(self, y, xVal, xInd):
        _sentry_ndim(1, xVal=xVal, xInd=xInd, y=y)
        _sentry_dtype(np.int32, xInd=xInd)
        _sentry_dtype(xVal.dtype, y=y)
        fn = self._get_api("gthr", xVal.dtype)
        nnz = xVal.size
        with _readonly(y, xInd) as [dy, dxind]:
            with _readwrite(xVal) as [dxval]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, idxBase=self.idxbase)

    def gthrz(self, y, xVal, xInd):
        _sentry_ndim(1, xVal=xVal, xInd=xInd, y=y)
        _sentry_dtype(np.int32, xInd=xInd)
        _sentry_dtype(xVal.dtype, y=y)
        fn = self._get_api("gthrz", xVal.dtype)
        nnz = xVal.size
        with _readonly(xInd) as [dxind]:
            with _readwrite(y, xVal) as [dy, dxval]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, idxBase=self.idxbase)

    def roti(self, xVal, xInd, y, c, s):
        _sentry_ndim(1, xVal=xVal, xInd=xInd, y=y)
        _sentry_dtype(np.int32, xInd=xInd)
        _sentry_dtype(xVal.dtype, y=y)
        fn = self._get_api("roti", xVal.dtype)
        nnz = xVal.size
        with _readonly(xInd) as [dxind]:
            with _readwrite(y, xVal) as [dy, dxval]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, c=c, s=s,
                   idxBase=self.idxbase)

    def sctr(self, xVal, xInd, y):
        _sentry_ndim(1, xVal=xVal, xInd=xInd, y=y)
        _sentry_dtype(np.int32, xInd=xInd)
        _sentry_dtype(xVal.dtype, y=y)
        fn = self._get_api("sctr", xVal.dtype)
        nnz = xVal.size
        with _readonly(xVal, xInd) as [dxval, dxind]:
            with _readwrite(y) as [dy]:
                fn(nnz=nnz, xVal=dxval, xInd=dxind, y=dy, idxBase=self.idxbase)

    # ------------------------------------------------------------------------
    # Level 2 API

    def bsrmv_matrix(self, dir, trans, alpha, descr, bsrmat, x, beta, y):
        bsrVal = bsrmat.data
        bsrRowPtr = bsrmat.indptr
        bsrColInd = bsrmat.indices
        nnzb = bsrColInd.size
        m, n = bsrmat.shape
        blockDim, blockDim1 = bsrmat.blocksize
        assert blockDim == blockDim1

        mb = (m + blockDim - 1) // blockDim
        nb = (n + blockDim - 1) // blockDim

        self.bsrmv(dir, trans, mb, nb, nnzb, alpha, descr, bsrVal,
                   bsrRowPtr, bsrColInd, blockDim, x, beta, y)

    def bsrmv(self, dir, trans, mb, nb, nnzb, alpha, descr, bsrVal,
              bsrRowPtr, bsrColInd, blockDim, x, beta, y):
        _sentry_ndim(1, x=x, y=y)
        _sentry_dtype(bsrVal.dtype, x=x, y=y)
        fn = self._get_api("bsrmv", bsrVal.dtype)

        with _readonly(bsrVal, bsrRowPtr, bsrColInd, x) \
            as [dbsrVal, dbsrRowPtr, dbsrColInd, dx]:
            with _readwrite(y) as [dy]:
                fn(dirA=dir, transA=trans, mb=mb, nb=nb, nnzb=nnzb,
                   alpha=alpha, descrA=descr, bsrValA=dbsrVal,
                   bsrRowPtrA=dbsrRowPtr, bsrColIndA=dbsrColInd,
                   blockDim=blockDim, x=dx, beta=beta, y=dy)

    def bsrxmv(self, dir, trans, sizeOfMask, mb, nb, nnzb, alpha, descr,
               bsrVal, bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd, blockDim,
               x, beta, y):
        _sentry_ndim(1, x=x, y=y)
        _sentry_dtype(bsrVal.dtype, x=x, y=y)
        fn = self._get_api("bsrxmv", bsrVal.dtype)

        with _readonly(bsrVal, bsrRowPtr, bsrColInd, bsrMaskPtr, bsrEndPtr, x) \
            as [dbsrVal, dbsrRowPtr, dbsrColInd, dbsrMaskPtr, dbsrEndPtr, dx]:
            with _readwrite(y) as [dy]:
                fn(dirA=dir, transA=trans, sizeOfMask=sizeOfMask,
                   mb=mb, nb=nb, nnzb=nnzb, alpha=alpha, descrA=descr,
                   bsrValA=dbsrVal, bsrRowPtrA=dbsrRowPtr,
                   bsrColIndA=dbsrColInd, bsrMaskPtrA=dbsrMaskPtr,
                   bsrEndPtrA=dbsrEndPtr, blockDim=blockDim, x=dx, beta=beta,
                   y=dy)

    def csrmv(self, trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr,
              csrColInd, x, beta, y):
        _sentry_ndim(1, x=x, y=y)
        _sentry_dtype(csrVal.dtype, x=x, y=y)
        fn = self._get_api("csrmv", csrVal.dtype)
        with _readonly(csrVal, csrRowPtr, csrColInd, x) \
            as [dcsrVal, dcsrRowPtr, dcsrColInd, dx]:
            with _readwrite(y) as [dy]:
                fn(transA=trans, m=m, n=n, nnz=nnz,
                   alpha=alpha, descrA=descr, csrValA=dcsrVal,
                   csrRowPtrA=dcsrRowPtr, csrColIndA=dcsrColInd, x=dx,
                   beta=beta, y=dy)

    def csrsv_analysis(self, trans, m, nnz, descr, csrVal, csrRowPtr,
                       csrColInd):
        """
        Returns
        -------
        SolveAnalysisInfo
        """
        fn = self._get_api("csrsv_analysis", csrVal.dtype)
        info = self.api.solve_analysis_info()
        with _readonly(csrVal, csrRowPtr, csrColInd) \
            as [dcsrVal, dcsrRowPtr, dcsrColInd]:
            fn(transA=trans, m=m, nnz=nnz, descrA=descr, csrValA=dcsrVal,
               csrRowPtrA=dcsrRowPtr, csrColIndA=dcsrColInd, info=info)
        return info

    def csrsv_solve(self, trans, m, alpha, descr, csrVal, csrRowPtr,
                    csrColInd, info, x, y):
        _sentry_ndim(1, x=x, y=y)
        _sentry_dtype(csrVal.dtype, x=x, y=y)
        fn = self._get_api("csrsv_solve", csrVal.dtype)
        with _readonly(csrVal, csrRowPtr, csrColInd, x) \
            as [dcsrVal, dcsrRowPtr, dcsrColInd, dx]:
            with _readwrite(y) as [dy]:
                fn(transA=trans, m=m, alpha=alpha, descrA=descr,
                   csrValA=dcsrVal, csrRowPtrA=dcsrRowPtr,
                   csrColIndA=dcsrColInd, info=info, x=dx, y=dy)

    hybmv = NotImplemented
    hybmv_analysis = NotImplemented
    hybmv_solve = NotImplemented

    # ------------------------------------------------------------------------
    # Level 3 API

    def csrmm(self, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA,
              csrColIndA, B, ldb, beta, C, ldc):
        _sentry_dtype(csrValA.dtype, B=B, C=C)
        fn = self._get_api("csrmm", csrValA.dtype)
        with _readonly(csrValA, csrRowPtrA, csrColIndA, B) \
            as [dcsrValA, dcsrRowPtrA, dcsrColIndA, dB]:
            with _readwrite(C) as [dC]:
                fn(transA=transA, m=m, n=n, k=k, nnz=nnz, alpha=alpha,
                   descrA=descrA, csrValA=dcsrValA, csrRowPtrA=dcsrRowPtrA,
                   csrColIndA=dcsrColIndA, B=dB, ldb=ldb, beta=beta, C=dC,
                   ldc=ldc)

    def csrmm2(self, transA, transB, m, n, k, nnz, alpha, descrA, csrValA,
               csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc):
        _sentry_dtype(csrValA.dtype, B=B, C=C)
        fn = self._get_api("csrmm2", csrValA.dtype)
        with _readonly(csrValA, csrRowPtrA, csrColIndA, B) \
            as [dcsrValA, dcsrRowPtrA, dcsrColIndA, dB]:
            with _readwrite(C) as [dC]:
                fn(transa=transA, transb=transB, m=m, n=n, k=k, nnz=nnz,
                   alpha=alpha,
                   descrA=descrA, csrValA=dcsrValA, csrRowPtrA=dcsrRowPtrA,
                   csrColIndA=dcsrColIndA, B=dB, ldb=ldb, beta=beta, C=dC,
                   ldc=ldc)

    def csrsm_analysis(self, transA, m, nnz, descrA, csrValA, csrRowPtrA,
                       csrColIndA):
        fn = self._get_api("csrsm_analysis", csrValA.dtype)
        info = self.api.solve_analysis_info()
        with _readonly(csrValA, csrRowPtrA, csrColIndA) \
            as [dcsrValA, dcsrRowPtrA, dcsrColIndA]:
            fn(transA=transA, m=m, nnz=nnz, descrA=descrA, csrValA=dcsrValA,
               csrRowPtrA=dcsrRowPtrA, csrColIndA=dcsrColIndA, info=info)
        return info

    def csrsm_solve(self, transA, m, n, alpha, descrA, csrValA, csrRowPtrA,
                    csrColIndA, info, X, ldx, Y, ldy):
        fn = self._get_api("csrsm_solve", csrValA.dtype)
        with _readonly(csrValA, csrRowPtrA, csrColIndA, X) \
            as [dcsrValA, dcsrRowPtrA, dcsrColIndA, dX]:
            with _readwrite(Y) as [dY]:
                fn(transA=transA, m=m, n=n, alpha=alpha, descrA=descrA,
                   csrValA=dcsrValA, csrRowPtrA=dcsrRowPtrA,
                   csrColIndA=dcsrColIndA, info=info, x=dX, ldx=ldx, y=dY,
                   ldy=ldy)

    def XcsrgeamNnz(self, m, n, descrA, nnzA, csrRowPtrA, csrColIndA, descrB,
                    nnzB, csrRowPtrB, csrColIndB, descrC, csrRowPtrC):
        """
        Returns
        -------
        nnzC
        """
        fn = self.api.XcsrgeamNnz
        with _readonly(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB) \
            as (dcsrRowPtrA, dcsrColIndA, dcsrRowPtrB, dcsrColIndB):
            with _readwrite(csrRowPtrC) as [dcsrRowPtrC]:
                nnzC = fn(m=m, n=n, descrA=descrA, nnzA=nnzA,
                          csrRowPtrA=dcsrRowPtrA,
                          csrColIndA=dcsrColIndA, descrB=descrB, nnzB=nnzB,
                          csrRowPtrB=dcsrRowPtrB, csrColIndB=dcsrColIndB,
                          descrC=descrC, csrRowPtrC=dcsrRowPtrC,
                          nnzTotalDevHostPtr=0)
        return nnzC

    def csrgeam(self, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA,
                csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB,
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC):
        fn = self._get_api("csrgeam", csrValA.dtype)
        with _readonly(csrValA, csrRowPtrA, csrColIndA, csrValB, csrRowPtrB,
                       csrColIndB, csrRowPtrC) \
            as [dcsrValA, dcsrRowPtrA, dcsrColIndA, dcsrValB, dcsrRowPtrB,
                dcsrColIndB, dcsrRowPtrC]:
            with _readwrite(csrValC, csrColIndC) as [dcsrValC, dcsrColIndC]:
                fn(m=m, n=n, alpha=alpha, descrA=descrA, nnzA=nnzA,
                   csrValA=dcsrValA, csrRowPtrA=dcsrRowPtrA,
                   csrColIndA=dcsrColIndA, csrValB=dcsrValB,
                   descrB=descrB, nnzB=nnzB, beta=beta,
                   csrRowPtrB=dcsrRowPtrB, csrColIndB=dcsrColIndB,
                   descrC=descrC, csrValC=dcsrValC,
                   csrRowPtrC=dcsrRowPtrC, csrColIndC=dcsrColIndC)


    def XcsrgemmNnz(self, transA, transB, m, n, k, descrA, nnzA, csrRowPtrA,
                    csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC,
                    csrRowPtrC):
        """
        Returns
        -------
        nnzC
        """
        fn = self.api.XcsrgemmNnz
        with _readonly(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB) \
            as (dcsrRowPtrA, dcsrColIndA, dcsrRowPtrB, dcsrColIndB):
            with _readwrite(csrRowPtrC) as [dcsrRowPtrC]:
                nnzC = fn(transA=transA, transB=transB, k=k, m=m, n=n,
                          descrA=descrA, nnzA=nnzA,
                          csrRowPtrA=dcsrRowPtrA,
                          csrColIndA=dcsrColIndA, descrB=descrB, nnzB=nnzB,
                          csrRowPtrB=dcsrRowPtrB, csrColIndB=dcsrColIndB,
                          descrC=descrC, csrRowPtrC=dcsrRowPtrC,
                          nnzTotalDevHostPtr=0)
        return nnzC

    def csrgemm(self, transA, transB, m, n, k, descrA, nnzA, csrValA,
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB,
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC):
        fn = self._get_api("csrgemm", csrValA.dtype)
        with _readonly(csrValA, csrRowPtrA, csrColIndA, csrValB, csrRowPtrB,
                       csrColIndB, csrRowPtrC) \
            as [dcsrValA, dcsrRowPtrA, dcsrColIndA, dcsrValB, dcsrRowPtrB,
                dcsrColIndB, dcsrRowPtrC]:
            with _readwrite(csrValC, csrColIndC) as [dcsrValC, dcsrColIndC]:
                fn(transA=transA, transB=transB, m=m, n=n, k=k, descrA=descrA,
                   nnzA=nnzA, csrValA=dcsrValA, csrRowPtrA=dcsrRowPtrA,
                   csrColIndA=dcsrColIndA, csrValB=dcsrValB,
                   descrB=descrB, nnzB=nnzB,
                   csrRowPtrB=dcsrRowPtrB, csrColIndB=dcsrColIndB,
                   descrC=descrC, csrValC=dcsrValC,
                   csrRowPtrC=dcsrRowPtrC, csrColIndC=dcsrColIndC)

# ------------------------------------------------------------------------
# Matrix Ctors


def bsr_matrix(*args, **kws):
    mat = ss.bsr_matrix(*args, **kws)
    return BSRCudaMatrix(mat)


class BSRCudaMatrix(object):
    def __init__(self, bsrmat, stream=0):
        self.dtype = bsrmat.dtype
        self.shape = bsrmat.shape
        self.ndim = bsrmat.ndim
        self.nnz = bsrmat.nnz
        self.data = cuda.to_device(bsrmat.data, stream=stream)
        self.indices = cuda.to_device(bsrmat.indices, stream=stream)
        self.indptr = cuda.to_device(bsrmat.indptr, stream=stream)
        self.blocksize = bsrmat.blocksize

    def copy_to_host(self, stream=0):
        data = self.data.copy_to_host(stream=stream)
        indices = self.indices.copy_to_host(stream=stream)
        indptr = self.indptr.copy_to_host(stream=stream)
        return ss.bsr_matrix((data, indices, indptr))
