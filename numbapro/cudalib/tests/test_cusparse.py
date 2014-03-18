import numpy as np
import scipy.sparse
import unittest
from .support import addtest, main
from numbapro.cudalib import cusparse
from numbapro.cudalib.cusparse.binding import CuSparseError
from numbapro import cuda


@addtest
class TestCuSparseLevel1(unittest.TestCase):
    def setUp(self):
        self.cus = cusparse.Sparse()

    def generic_test_axpyi(self, dtype):
        alpha = 2
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.zeros(shape=xval.size * 2, dtype=xval.dtype)
        self.cus.axpyi(alpha, xval, xind, y)
        self.assertTrue(np.allclose(y[xind], (xval * 2)))

    def test_Saxpyi(self):
        self.generic_test_axpyi(dtype=np.float32)

    def test_Daxpyi(self):
        self.generic_test_axpyi(dtype=np.float64)

    def test_Caxpyi(self):
        self.generic_test_axpyi(dtype=np.complex64)

    def test_Zaxpyi(self):
        self.generic_test_axpyi(dtype=np.complex128)

    def generic_test_doti(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        result = self.cus.doti(xval, xind, y)
        self.assertTrue(result)

    def test_Sdoti(self):
        self.generic_test_doti(dtype=np.float32)

    def test_Zdoti(self):
        self.generic_test_doti(dtype=np.complex128)

    def generic_test_dotci(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        result = self.cus.dotci(xval, xind, y)
        self.assertTrue(result)

    def test_Zdotci(self):
        self.generic_test_dotci(dtype=np.complex128)

    def generic_test_gthr(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        self.cus.gthr(y, xval, xind)
        self.assertTrue(np.all(xval == 1))

    def test_Sgthr(self):
        self.generic_test_gthr(dtype=np.float32)

    def test_Cgthr(self):
        self.generic_test_gthr(dtype=np.complex64)

    def generic_test_gthrz(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        self.cus.gthrz(y, xval, xind)
        self.assertTrue(np.all(xval == 1))
        self.assertTrue(np.all(y[xind] == 0))

    def test_Dgthr(self):
        self.generic_test_gthrz(dtype=np.float64)

    def test_Zgthr(self):
        self.generic_test_gthrz(dtype=np.complex128)

    def generic_test_roti(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        c = .2
        s = .3
        oldxval = xval.copy()
        oldy = y.copy()
        self.cus.roti(xval, xind, y, c, s)
        self.assertFalse(np.all(oldxval == xval))
        self.assertFalse(np.all(oldy == y))

    def test_Sroti(self):
        self.generic_test_roti(dtype=np.float32)

    def test_Droti(self):
        self.generic_test_roti(dtype=np.float64)

    def generic_test_sctr(self, dtype):
        xval = np.arange(5, dtype=dtype) + 1
        xind = np.arange(xval.size, dtype='int32') * 2
        y = np.ones(shape=xval.size * 2, dtype=xval.dtype)
        oldy = y.copy()
        self.cus.sctr(xval, xind, y)
        self.assertFalse(np.all(oldy == y))

    def test_Ssctr(self):
        self.generic_test_sctr(dtype=np.float32)

    def test_Csctr(self):
        self.generic_test_sctr(dtype=np.complex64)


@addtest
class TestCuSparseMatrixOp(unittest.TestCase):
    def test_bsr_matrix(self):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        expect = scipy.sparse.bsr_matrix((data, (row, col)), shape=(3, 3))
        mat = cusparse.bsr_matrix((data, (row, col)), shape=(3, 3))
        host = mat.copy_to_host()
        self.assertTrue(np.all(host.indices == expect.indices))
        self.assertTrue(np.all(host.indptr == expect.indptr))
        self.assertTrue(np.all(host.data == expect.data))

    def test_matdescr(self):
        sparse = cusparse.Sparse()
        md = sparse.matdescr()
        md.diagtype = 'N'
        md.fillmode = 'L'
        md.indexbase = 0
        md.matrixtype = 'G'

        self.assertEqual('N', md.diagtype)
        self.assertEqual('L', md.fillmode)
        self.assertEqual(0, md.indexbase)
        self.assertEqual('G', md.matrixtype)
        del md


@addtest
class TestCuSparseLevel2(unittest.TestCase):
    def setUp(self):
        self.cus = cusparse.Sparse()

    def generic_test_bsrmv(self, dtype):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)

        bsrmat = cusparse.bsr_matrix((data, (row, col)), shape=(3, 3))
        x = np.ones(3, dtype=dtype)
        y = np.ones(3, dtype=dtype)
        oldy = y.copy()

        alpha = 1
        beta = 1
        descr = self.cus.matdescr()
        self.cus.bsrmv_matrix('C', 'N', alpha, descr, bsrmat, x, beta, y)

        self.assertFalse(np.all(y == oldy))

    def test_Sbsrmv(self):
        dtype = np.float32
        self.generic_test_bsrmv(dtype=dtype)

    def test_Cbsrmv(self):
        dtype = np.complex64
        self.generic_test_bsrmv(dtype=dtype)

    def test_Sbsrxmv(self):
        """
        Just exercise the codepath
        """
        dtype = np.float32
        alpha = 0
        beta = 0
        descr = self.cus.matdescr()
        bsrVal = np.zeros(10, dtype=dtype)
        bsrMaskPtr = np.zeros(10, dtype=np.int32)
        bsrRowPtr = np.zeros(10, dtype=np.int32)
        bsrEndPtr = np.zeros(10, dtype=np.int32)
        bsrColInd = np.zeros(10, dtype=np.int32)
        blockDim = 1
        x = np.zeros(10, dtype=dtype)
        y = np.zeros(10, dtype=dtype)
        self.cus.bsrxmv('C', 'N', 1, 1, 1, 1, alpha, descr, bsrVal,
                        bsrMaskPtr, bsrRowPtr, bsrEndPtr, bsrColInd,
                        blockDim, x, beta, y)

    def test_Scsrmv(self):
        """
        Just exercise the codepath
        """
        dtype = np.float32
        alpha = 0
        beta = 0
        descr = self.cus.matdescr()
        csrVal = np.zeros(10, dtype=dtype)
        csrColInd = np.zeros(10, dtype=np.int32)
        csrRowPtr = np.zeros(10, dtype=np.int32)
        x = np.zeros(10, dtype=dtype)
        y = np.zeros(10, dtype=dtype)
        trans = 'N'
        m = 1
        n = 1
        nnz = 1
        self.cus.csrmv(trans, m, n, nnz, alpha, descr, csrVal, csrRowPtr,
                       csrColInd, x, beta, y)

    def test_Scsrmv(self):
        """
        Just exercise the codepath
        """
        dtype = np.float32

        descr = self.cus.matdescr()
        csrVal = np.zeros(10, dtype=dtype)
        csrColInd = np.zeros(10, dtype=np.int32)
        csrRowPtr = np.zeros(10, dtype=np.int32)
        trans = 'N'
        m = 1
        nnz = 1
        info = self.cus.csrsv_analysis(trans, m, nnz, descr, csrVal,
                                       csrRowPtr, csrColInd)

        alpha = 1.0
        x = np.zeros(10, dtype=dtype)
        y = np.zeros(10, dtype=dtype)
        try:
            self.cus.csrsv_solve(trans, m, alpha, descr, csrVal, csrRowPtr,
                                 csrColInd, info, x, y)
        except CuSparseError:
            pass


if __name__ == '__main__':
    main()

