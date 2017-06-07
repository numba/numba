import numpy as np
from numba import ocl
from numba.ocl.testing import unittest


class TestArrayAttr(unittest.TestCase):
    def test_contigous_2d(self):
        ary = np.arange(10)
        cary = ary.reshape(2, 5)
        fary = np.asfortranarray(cary)

        dcary = ocl.to_device(cary)
        dfary = ocl.to_device(fary)
        self.assertTrue(dcary.is_c_contiguous())
        self.assertTrue(not dfary.is_c_contiguous())
        self.assertTrue(not dcary.is_f_contiguous())
        self.assertTrue(dfary.is_f_contiguous())

    def test_contigous_3d(self):
        ary = np.arange(20)
        cary = ary.reshape(2, 5, 2)
        fary = np.asfortranarray(cary)

        dcary = ocl.to_device(cary)
        dfary = ocl.to_device(fary)
        self.assertTrue(dcary.is_c_contiguous())
        self.assertTrue(not dfary.is_c_contiguous())
        self.assertTrue(not dcary.is_f_contiguous())
        self.assertTrue(dfary.is_f_contiguous())

    def test_contigous_4d(self):
        ary = np.arange(60)
        cary = ary.reshape(2, 5, 2, 3)
        fary = np.asfortranarray(cary)

        dcary = ocl.to_device(cary)
        dfary = ocl.to_device(fary)
        self.assertTrue(dcary.is_c_contiguous())
        self.assertTrue(not dfary.is_c_contiguous())
        self.assertTrue(not dcary.is_f_contiguous())
        self.assertTrue(dfary.is_f_contiguous())

    def test_ravel_c(self):
        ary = np.arange(60)
        reshaped = ary.reshape(2, 5, 2, 3)
        expect = reshaped.ravel(order='C')
        dary = ocl.to_device(reshaped)
        dflat = dary.ravel()
        flat = dflat.copy_to_host()
        self.assertTrue(flat.ndim == 1)
        self.assertTrue(np.all(expect == flat))

    def test_ravel_f(self):
        ary = np.arange(60)
        reshaped = np.asfortranarray(ary.reshape(2, 5, 2, 3))
        expect = reshaped.ravel(order='F')
        dary = ocl.to_device(reshaped)
        dflat = dary.ravel(order='F')
        flat = dflat.copy_to_host()
        self.assertTrue(flat.ndim == 1)
        self.assertTrue(np.all(expect == flat))

    def test_reshape_c(self):
        ary = np.arange(10)
        expect = ary.reshape(2, 5)
        dary = ocl.to_device(ary)
        dary_reshaped = dary.reshape(2, 5)
        got = dary_reshaped.copy_to_host()
        self.assertTrue(np.all(expect == got))

    def test_reshape_f(self):
        ary = np.arange(10)
        expect = ary.reshape(2, 5, order='F')
        dary = ocl.to_device(ary)
        dary_reshaped = dary.reshape(2, 5, order='F')
        got = dary_reshaped.copy_to_host()
        self.assertTrue(np.all(expect == got))


if __name__ == '__main__':
    unittest.main()

