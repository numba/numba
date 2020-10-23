import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support


def set_a(ary, i, v):
    ary[i].a = v


def set_b(ary, i, v):
    ary[i].b = v


def set_c(ary, i, v):
    ary[i].c = v


def set_record(ary, i, j):
    ary[i] = ary[j]


def record_set_a(r, v):
    r.a = v


def record_set_b(r, v):
    r.b = v


def record_set_c(r, v):
    r.c = v


def record_read_a(r, arr):
    arr[0] = r.a


def record_read_b(r, arr):
    arr[0] = r.b


def record_read_c(r, arr):
    arr[0] = r.c


def record_write_array(r):
    r.g = 2
    r.h[0] = 3.0
    r.h[1] = 4.0


def record_write_2d_array(r):
    r.i = 3
    r.j[0, 0] = 5.0
    r.j[0, 1] = 6.0
    r.j[1, 0] = 7.0
    r.j[1, 1] = 8.0
    r.j[2, 0] = 9.0
    r.j[2, 1] = 10.0


def record_read_array(r, a):
    a[0] = r.h[0]
    a[1] = r.h[1]


def record_read_2d_array(r, a):
    a[0, 0] = r.j[0, 0]
    a[0, 1] = r.j[0, 1]
    a[1, 0] = r.j[1, 0]
    a[1, 1] = r.j[1, 1]
    a[2, 0] = r.j[2, 0]
    a[2, 1] = r.j[2, 1]


recordtype = np.dtype(
    [
        ('a', np.float64),
        ('b', np.int32),
        ('c', np.complex64),
        ('d', (np.str, 5))
    ],
    align=True
)

recordwitharray = np.dtype(
    [
        ('g', np.int32),
        ('h', np.float32, 2)
    ],
    align=True
)

recordwith2darray = np.dtype([('i', np.int32),
                              ('j', np.float32, (3, 2))])


class TestRecordDtype(CUDATestCase):

    def _createSampleArrays(self):
        self.sample1d = np.recarray(3, dtype=recordtype)
        self.samplerec1darr = np.recarray(1, dtype=recordwitharray)[0]
        self.samplerec2darr = np.recarray(1, dtype=recordwith2darray)[0]

    def setUp(self):
        self._createSampleArrays()

        ary = self.sample1d
        for i in range(ary.size):
            x = i + 1
            ary[i]['a'] = x / 2
            ary[i]['b'] = x
            ary[i]['c'] = x * 1j
            ary[i]['d'] = "%d" % x

    def get_cfunc(self, pyfunc, argspec):
        return cuda.jit()(pyfunc)

    def _test_set_equal(self, pyfunc, value, valuetype):
        rec = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (rec[:], types.intp, valuetype))

        for i in range(self.sample1d.size):
            got = self.sample1d.copy()

            # Force the argument to the pure Python function to be
            # a recarray, as attribute access isn't supported on
            # structured arrays.
            expect = got.copy().view(np.recarray)

            cfunc[1, 1](got, i, value)
            pyfunc(expect, i, value)

            # Match the entire array to ensure no memory corruption
            self.assertTrue(np.all(expect == got))

    def test_set_a(self):
        self._test_set_equal(set_a, 3.1415, types.float64)
        # Test again to check if coercion works
        self._test_set_equal(set_a, 3., types.float32)

    def test_set_b(self):
        self._test_set_equal(set_b, 123, types.int32)
        # Test again to check if coercion works
        self._test_set_equal(set_b, 123, types.float64)

    def test_set_c(self):
        self._test_set_equal(set_c, 43j, types.complex64)
        # Test again to check if coercion works
        self._test_set_equal(set_c, 43j, types.complex128)

    def test_set_record(self):
        pyfunc = set_record
        rec = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (rec[:], types.intp, types.intp))

        test_indices = [(0, 1), (1, 2), (0, 2)]
        for i, j in test_indices:
            expect = self.sample1d.copy()
            pyfunc(expect, i, j)

            got = self.sample1d.copy()
            cfunc[1, 1](got, i, j)

            # Match the entire array to ensure no memory corruption
            self.assertEqual(expect[i], expect[j])
            self.assertEqual(got[i], got[j])
            self.assertTrue(np.all(expect == got))

    def _test_rec_set(self, v, pyfunc, f):
        rec = self.sample1d.copy()[0]
        nbrecord = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (nbrecord,))
        cfunc[1, 1](rec, v)
        np.testing.assert_equal(rec[f], v)

    def test_rec_set_a(self):
        self._test_rec_set(np.float64(1.5), record_set_a, 'a')

    def test_rec_set_b(self):
        self._test_rec_set(np.int32(2), record_set_b, 'b')

    def test_rec_set_c(self):
        self._test_rec_set(np.complex64(4.0 + 5.0j), record_set_c, 'c')

    def _test_rec_read(self, v, pyfunc, f):
        rec = self.sample1d.copy()[0]
        rec[f] = v
        arr = np.zeros(1, v.dtype)
        nbrecord = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (nbrecord,))
        cfunc[1, 1](rec, arr)
        np.testing.assert_equal(arr[0], v)

    def test_rec_read_a(self):
        self._test_rec_read(np.float64(1.5), record_read_a, 'a')

    def test_rec_read_b(self):
        self._test_rec_read(np.int32(2), record_read_b, 'b')

    def test_rec_read_c(self):
        self._test_rec_read(np.complex64(4.0 + 5.0j), record_read_c, 'c')

    def test_record_write_1d_array(self):
        '''
        Test writing to a 1D array within a structured type
        '''
        rec = self.samplerec1darr.copy()
        nbrecord = numpy_support.from_dtype(recordwitharray)
        cfunc = self.get_cfunc(record_write_array, (nbrecord,))

        cfunc[1, 1](rec)
        expected = self.samplerec1darr.copy()
        expected['g'] = 2
        expected['h'][0] = 3.0
        expected['h'][1] = 4.0

        np.testing.assert_equal(expected, rec)

    def test_record_write_2d_array(self):
        '''
        Test writing to a 2D array within a structured type
        '''
        rec = self.samplerec2darr.copy()
        nbrecord = numpy_support.from_dtype(recordwith2darray)
        cfunc = self.get_cfunc(record_write_2d_array, (nbrecord,))
        cfunc[1, 1](rec)

        expected = self.samplerec2darr.copy()
        expected['i'] = 3
        expected['j'][:] = np.asarray([5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                                      np.float32).reshape(3, 2)
        np.testing.assert_equal(expected, rec)

    def test_record_read_1d_array(self):
        '''
        Test reading from a 1D array within a structured type
        '''
        rec = self.samplerec1darr.copy()
        rec['h'][0] = 4.0
        rec['h'][1] = 5.0

        nbrecord = numpy_support.from_dtype(recordwitharray)
        cfunc = self.get_cfunc(record_read_array, (nbrecord,))
        arr = np.zeros(2, dtype=rec['h'].dtype)
        cfunc[1, 1](rec, arr)

        np.testing.assert_equal(rec['h'], arr)

    def test_record_read_2d_array(self):
        '''
        Test reading from a 2D array within a structured type
        '''
        rec = self.samplerec2darr.copy()
        rec['j'][:] = np.asarray([5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                                 np.float32).reshape(3, 2)

        nbrecord = numpy_support.from_dtype(recordwith2darray)
        cfunc = self.get_cfunc(record_read_2d_array, (nbrecord,))
        arr = np.zeros((3,2), dtype=rec['j'].dtype)
        cfunc[1, 1](rec, arr)

        np.testing.assert_equal(rec['j'], arr)


@skip_on_cudasim('Structured array attr access not supported in simulator')
class TestRecordDtypeWithStructArrays(TestRecordDtype):
    '''
    Same as TestRecordDtype, but using structured arrays instead of recarrays.
    '''

    def _createSampleArrays(self):
        self.sample1d = np.zeros(3, dtype=recordtype)
        self.samplerec1darr = np.zeros(1, dtype=recordwitharray)[0]
        self.samplerec2darr = np.zeros(1, dtype=recordwith2darray)[0]


if __name__ == '__main__':
    unittest.main()
