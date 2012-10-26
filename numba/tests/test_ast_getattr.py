#! /usr/bin/env python
# ______________________________________________________________________

from numba.translate import _plat_bits
from numba.decorators import autojit

import numpy as np
import numpy

import unittest

# ______________________________________________________________________

def _get_ndarray_ndim(ndarr):
    return ndarr.ndim

def _get_ndarray_shape(ndarr):
    return ndarr.shape

def _get_ndarray_data(ndarr):
    return ndarr.data

def _get_ndarray_2_shape_unpack_0(ndarr):
    dim0, _ = ndarr.shape
    return dim0

def _get_ndarray_2_shape_unpack_1(ndarr):
    _, dim1 = ndarr.shape
    return dim1

get_ndarray_ndim = autojit(backend='ast')(_get_ndarray_ndim)
get_ndarray_shape = autojit(backend='ast')(_get_ndarray_shape)
get_ndarray_data = autojit(backend='ast')(_get_ndarray_data)
get_ndarray_2_shape_unpack_0 = autojit(backend='ast')(_get_ndarray_2_shape_unpack_0)
get_ndarray_2_shape_unpack_1 = autojit(backend='ast')(_get_ndarray_2_shape_unpack_1)

# ______________________________________________________________________

class TestGetattr(unittest.TestCase):
    def test_getattr_ndim(self):
        args = [
            np.empty((2,)),
            np.empty((2, 2)),
        ]

        for arg in args:
            expect = _get_ndarray_ndim(arg)
            got = get_ndarray_ndim(arg)
            self.assertEqual(got, expect)

    def test_getattr_shape(self):
        args = [
            np.empty((10,)),
            np.empty((10, 20)),
        ]

        for arg in args:
            expect = _get_ndarray_shape(arg)
            got = get_ndarray_shape(arg)
            for i, _ in enumerate(expect):
                print i
                self.assertEqual(got[i], expect[i])

    def test_getattr_shape_unpack(self):
        args = [
            np.empty((1, 2))
        ]

        for arg in args:
            expect_dim0 = get_ndarray_2_shape_unpack_0(arg)
            expect_dim1 = get_ndarray_2_shape_unpack_1(arg)
            got_dim0 = _get_ndarray_2_shape_unpack_0(arg)
            got_dim1 = _get_ndarray_2_shape_unpack_1(arg)

            expect = expect_dim0, expect_dim1
            got = got_dim0, got_dim1

            self.assertEqual(got, expect)

    def test_getattr_data_1(self):
        expect = [1., 2., 3.]
        test_data = numpy.array([1., 2., 3.])
        got = get_ndarray_data(test_data)

        # this returns a buffer object
        #   _get_ndarray_data(test_data)
        for i, _ in enumerate(expect):
            self.assertEqual(got[i], expect[i])

    def test_getattr_data_2(self):
        expect = map(float, range(6))
        test_data = numpy.array(expect).reshape((2, 3))
        got = get_ndarray_data(test_data)
        for i, v in enumerate(expect):
            self.assertEqual(got[i], v)

# ______________________________________________________________________

if __name__ == "__main__":
#    TestGetattr('test_getattr_data_1').debug()
    unittest.main()

# ______________________________________________________________________
# End of test_ast_getattr.py
