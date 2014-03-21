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

    # ------------------------------------------------------------------------
    # Extra API

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

    def csrgemm_ez(self, matA, matB, transA='N', transB='N', descrA=None,
                       descrB=None, descrC=None):
        """
        Returns a csr matrix of the matrix product (matA * matB).

        Raises ValueError if the result is entirely zero.

        Notes
        -----
        Calls XcsrgemmNnz and csrgemm
        """
        tmpdescr = self.matdescr()
        descrA = descrA or tmpdescr
        descrB = descrB or tmpdescr
        descrC = descrC or tmpdescr

        dtype = matA.dtype
        m, ka = matA.shape
        kb, n = matB.shape
        if ka != kb:
            raise ValueError("incompatible matrices")
        k = ka

        indptrC = cuda.device_array(m + 1, dtype='int32')
        nnz = self.XcsrgemmNnz(transA, transB, m, n, k, descrA, matA.nnz,
                               matA.indptr, matA.indices, descrB, matB.nnz,
                               matB.indptr, matB.indices, descrC, indptrC)

        if nnz == 0:
            raise ValueError("result is entirely zero")

        dataC = cuda.device_array(nnz, dtype=dtype)
        indicesC = cuda.device_array(nnz, dtype='int32')
        self.csrgemm(transA, transB, m, n, k, descrA, matA.nnz, matA.data,
                     matA.indptr, matA.indices, descrB, matB.nnz, matB.data,
                     matB.indptr, matB.indices, descrC, dataC, indptrC,
                     indicesC)

        return CudaCSRMatrix().from_attributes(data=dataC, indices=indicesC,
                                               indptr=indptrC, shape=(m, n),
                                               dtype=dtype, nnz=nnz)

    # ------------------------------------------------------------------------
    # Preconditioners

    def csric0(self, trans, m, descr, csrValM, csrRowPtrA, csrColIndA, info):
        fn = self._get_api("csric0", csrValM.dtype)
        with _readonly(csrRowPtrA, csrColIndA) as [dcsrRowPtrA, dcsrColIndA]:
            with _readwrite(csrValM) as [dcsrValM]:
                fn(trans=trans, m=m, descrA=descr,
                   csrValA_ValM=dcsrValM, csrRowPtrA=dcsrRowPtrA,
                   csrColIndA=dcsrColIndA, info=info)

    def csrilu0(self, trans, m, descr, csrValM, csrRowPtrA, csrColIndA, info):
        fn = self._get_api("csrilu0", csrValM.dtype)
        with _readonly(csrRowPtrA, csrColIndA) as [dcsrRowPtrA, dcsrColIndA]:
            with _readwrite(csrValM) as [dcsrValM]:
                fn(trans=trans, m=m, descrA=descr,
                   csrValA_ValM=dcsrValM, csrRowPtrA=dcsrRowPtrA,
                   csrColIndA=dcsrColIndA, info=info)

    def gtsv(self, m, n, dl, d, du, B, ldb):
        fn = self._get_api("gtsv", B.dtype)
        with _readonly(dl, d, du) as [ddl, dd, ddu]:
            with _readwrite(B) as [dB]:
                fn(m=m, n=n, dl=ddl, d=dd, du=ddu, B=dB, ldb=ldb)

    def gtsv_nopivot(self, m, n, dl, d, du, B, ldb):
        fn = self._get_api("gtsv_nopivot", B.dtype)
        with _readonly(dl, d, du) as [ddl, dd, ddu]:
            with _readwrite(B) as [dB]:
                fn(m=m, n=n, dl=ddl, d=dd, du=ddu, B=dB, ldb=ldb)

    def gtsvStridedBatch(self, m, dl, d, du, x, batchCount, batchStride):
        fn = self._get_api("gtsvStridedBatch", x.dtype)
        with _readonly(dl, d, du) as [ddl, dd, ddu]:
            with _readwrite(x) as [dx]:
                fn(m=m, dl=ddl, d=dd, du=ddu, x=dx,
                   batchCount=batchCount, batchStride=batchStride)

    # ------------------------------------------------------------------------
    # Format Conversion

    def bsr2csr(self, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, bsrColIndA,
                blockDim, descrC, csrValC, csrRowPtrC, csrColIndC):
        fn = self._get_api('bsr2csr', bsrValA.dtype)
        with _readonly(bsrValA, bsrRowPtrA, bsrColIndA) as [dbsrValA,
                                                            dbsrRowPtrA,
                                                            dbsrColIndA]:
            with _readwrite(csrValC, csrRowPtrC, csrColIndC) as [dcsrValC,
                                                                 dcsrRowPtrC,
                                                                 dcsrColIndC]:
                fn(dirA=dirA, mb=mb, nb=nb, descrA=descrA, bsrValA=dbsrValA,
                   bsrRowPtrA=dbsrRowPtrA, bsrColIndA=dbsrColIndA,
                   blockDim=blockDim, descrC=descrC, csrValC=dcsrValC,
                   csrRowPtrC=dcsrRowPtrC, csrColIndC=dcsrColIndC)

    def Xcoo2csr(self, cooRowInd, nnz, m, csrRowPtr):
        fn = self.api.Xcoo2csr
        with _readonly(cooRowInd) as [dcooRowInd]:
            with _readwrite(csrRowPtr) as [dcsrRowPtr]:
                fn(cooRowInd=dcooRowInd, nnz=nnz, m=m, csrRowPtr=dcsrRowPtr,
                   idxBase=self.idxbase)

    def csc2dense(self, m, n, descrA, cscValA, cscRowIndA, cscColPtrA, A, lda):
        fn = self._get_api('csc2dense', cscValA.dtype)
        with _readonly(cscValA, cscRowIndA, cscColPtrA) as [dcscValA,
                                                            dcscRowIndA,
                                                            dcscColPtrA]:
            with _readwrite(A) as [dA]:
                fn(m=m, n=n, descrA=descrA, cscValA=dcscValA,
                   cscRowIndA=dcscRowIndA, cscColPtrA=dcscColPtrA, A=dA,
                   lda=lda)

    csc2hyb = NotImplemented

    def Xcsr2bsrNnz(self, dirA, m, n, descrA, csrRowPtrA, csrColIndA,
                    blockDim, descrC, bsrRowPtrC):
        fn = self.api.Xcsr2bsrNnz
        with _readonly(csrRowPtrA, csrColIndA, bsrRowPtrC) as [dcsrRowPtrA,
                                                               dcsrColIndA,
                                                               dbsrRowPtrC]:
            nnz = fn(dirA=dirA, m=m, n=n, descrA=descrA,
                     csrRowPtrA=dcsrRowPtrA,
                     csrColIndA=dcsrColIndA,
                     blockDim=blockDim,
                     descrC=descrC, bsrRowPtrC=dbsrRowPtrC,
                     nnzTotalDevHostPtr=0)
        return nnz

    def csr2bsr(self, dirA, m, n, descrA, csrValA, csrRowPtrA, csrColIndA,
                blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC):
        fn = self._get_api('csr2bsr', csrValA.dtype)
        with _readonly(csrValA, csrRowPtrA, csrColIndA) as [dcsrValA,
                                                            dcsrRowPtrA,
                                                            dcsrColIndA]:
            with _readwrite(bsrValC, bsrRowPtrC, bsrColIndC) as [dbsrValC,
                                                                 dbsrRowPtrC,
                                                                 dbsrColIndC]:
                nnz = fn(dirA=dirA, m=m, n=n, descrA=descrA, csrValA=dcsrValA,
                         csrRowPtrA=dcsrRowPtrA, csrColIndA=dcsrColIndA,
                         blockDim=blockDim, descrC=descrC, bsrValC=dbsrValC,
                         bsrRowPtrC=dbsrRowPtrC, bsrColIndC=dbsrColIndC)
        return nnz

    def Xcsr2coo(self, csrRowPtr, nnz, m, cooRowInd):
        fn = self.api.Xcsr2coo
        with _readonly(csrRowPtr) as [dcsrRowPtr]:
            with _readwrite(cooRowInd) as [dcooRowInd]:
                fn(csrRowPtr=dcsrRowPtr, nnz=nnz, m=m, cooRowInd=dcooRowInd,
                   idxBase=self.idxbase)

    def csr2csc(self, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal,
                cscRowInd, cscColPtr, copyValues):
        fn = self._get_api('csr2csc', csrVal.dtype)
        with _readonly(csrVal, csrRowPtr, csrColInd) as [dcsrVal, dcsrRowPtr,
                                                         dcsrColInd]:
            with _readwrite(cscVal, cscRowInd, cscColPtr) as [dcscVal,
                                                              dcscRowInd,
                                                              dcscColPtr]:
                fn(m=m, n=n, nnz=nnz, csrVal=dcsrVal, csrRowPtr=dcsrRowPtr,
                   csrColInd=dcsrColInd, cscVal=dcscVal, cscRowInd=dcscRowInd,
                   cscColPtr=dcscColPtr, copyValues=copyValues,
                   idxBase=self.idxbase)

    def csr2dense(self, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda):
        fn = self._get_api('csr2dense', csrValA.dtype)
        with _readonly(csrValA, csrRowPtrA, csrColIndA) as [dcsrValA,
                                                            dcsrRowPtrA,
                                                            dcsrColIndA]:
            with _readwrite(A) as [dA]:
                fn(m=m, n=n, descrA=descrA, csrValA=dcsrValA,
                   csrRowPtrA=dcsrRowPtrA, csrColIndA=dcsrColIndA, A=dA,
                   lda=lda)

    csr2hyb = NotImplemented

    def dense2csc(self, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA,
                  cscColPtrA):
        fn = self._get_api('dense2csc', cscValA.dtype)
        with _readonly(A) as [dA]:
            with _readwrite(cscValA, cscRowIndA, cscColPtrA) as [dcscValA,
                                                                 dcscRowIndA,
                                                                 dcscColPtrA]:
                fn(m=m, n=n, descrA=descrA, A=dA, lda=lda,
                   nnzPerCol=nnzPerCol, cscValA=dcscValA,
                   cscRowIndA=dcscRowIndA,
                   cscColPtrA=dcscColPtrA)

    def dense2csr(self, m, n, descrA, A, lda, nnzPerRow, csrValA,
                  csrRowPtrA, csrColIndA):
        """
        Returns
        -------
        nnzTotalDevHostPtr
        """
        fn = self._get_api('dense2csr', A.dtype)
        with _readonly(A) as [dA]:
            with _readwrite(csrValA, csrRowPtrA, csrColIndA) as [dcsrValA,
                                                                 dcsrRowPtrA,
                                                                 dcsrColIndA]:
                fn(m=m, n=n, descrA=descrA, A=dA, lda=lda,
                   nnzPerRow=nnzPerRow, csrValA=dcsrValA,
                   csrRowPtrA=dcsrRowPtrA, csrColIndA=dcsrColIndA)

    dense2hyb = NotImplemented
    hyb2csc = NotImplemented
    hyb2csr = NotImplemented
    hyb2dense = NotImplemented

    def nnz(self, dirA, m, n, descrA, A, lda, nnzPerRowCol):
        fn = self._get_api('nnz', A.dtype)
        with _readonly(A) as [dA]:
            with _readwrite(nnzPerRowCol) as [dnnzPerRowCol]:
                nnzTotal = fn(dirA=dirA, m=m, n=n, descrA=descrA, A=dA,
                              nnzPerRowCol=dnnzPerRowCol, lda=lda,
                              nnzTotalDevHostPtr=0)
        return nnzTotal


# ------------------------------------------------------------------------
# Matrix Ctors

class CudaSparseMatrix(object):
    def from_host_matrix(self, matrix, stream=0):
        dtype = matrix.dtype
        shape = matrix.shape
        nnz = matrix.nnz
        data = cuda.to_device(matrix.data, stream=stream)
        indices = cuda.to_device(matrix.indices, stream=stream)
        indptr = cuda.to_device(matrix.indptr, stream=stream)
        self.from_attributes(dtype=dtype, shape=shape, nnz=nnz, data=data,
                             indices=indices, indptr=indptr)
        return self

    def from_attributes(self, dtype, shape, nnz, data, indices, indptr):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
        self.nnz = nnz
        self.data = data
        self.indices = indices
        self.indptr = indptr
        return self

    def copy_to_host(self, stream=0):
        data = self.data.copy_to_host(stream=stream)
        indices = self.indices.copy_to_host(stream=stream)
        indptr = self.indptr.copy_to_host(stream=stream)
        return self.host_constructor((data, indices, indptr), shape=self.shape)


class CudaBSRMatrix(CudaSparseMatrix):
    host_constructor = ss.bsr_matrix

    def from_host_matrix(self, matrix, stream=0):
        super(CudaBSRMatrix, self).from_host_matrix(matrix, stream=stream)
        self.blocksize = matrix.blocksize
        return self


class CudaCSCMatrix(CudaSparseMatrix):
    host_constructor = ss.csc_matrix


class CudaCSRMatrix(CudaSparseMatrix):
    host_constructor = ss.csr_matrix


def bsr_matrix(*args, **kws):
    mat = ss.bsr_matrix(*args, **kws)
    return CudaBSRMatrix().from_host_matrix(mat)


def csc_matrix(*args, **kws):
    mat = ss.csc_matrix(*args, **kws)
    return CudaCSCMatrix().from_host_matrix(mat)


def csr_matrix(*args, **kws):
    mat = ss.csr_matrix(*args, **kws)
    return CudaCSRMatrix().from_host_matrix(mat)
