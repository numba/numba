#! /usr/bin/env python
# ______________________________________________________________________

from numba.decorators import numba_compile

import numpy

import unittest

# ______________________________________________________________________

def get_ndarray_ndim(ndarr):
    return ndarr.ndim

def get_ndarray_shape(ndarr):
    return ndarr.shape

def get_ndarray_data(ndarr):
    return ndarr.data

# ______________________________________________________________________

class TestGetattr(unittest.TestCase):
    def test_getattr_ndim(self):
        test_data = numpy.array([1., 2., 3.])
        compiled_fn = numba_compile(ret_type = 'i',
                                    arg_types = [['d']])(get_ndarray_ndim)
        self.assertEqual(compiled_fn(test_data), 1)

    def test_getattr_shape(self):
        test_data = numpy.array([1., 2., 3.])
        compiled_fn = numba_compile(arg_types = [['d']])(get_ndarray_shape)
        result = compiled_fn(test_data)
        raise NotImplementedError('Need to determine the wrapped output type '
                                  'of int *')

    def test_getattr_data(self):
        test_data = numpy.array([1., 2., 3.])
        compiled_fn = numba_compile(arg_types = [['d']])(get_ndarray_data)
        result = compiled_fn(test_data)
        raise NotImplementedError('Need to determine the wrapped output type '
                                  'of double *')

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_getattr.py
