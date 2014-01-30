from __future__ import print_function
import numba.unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()


def create_array(control):
    return (np.array([1,2,3]) == control).all()

def create_empty_array(control):
    return (np.array([]) == control).all()

def create_arange(control):
    return (np.arange(10) == control).all()

def create_empty(control):
    my = np.empty(10)
    return (my.shape == control.shape and my.strides == control.strides and
            my.dtype == control.dtype)

def create_ones(control):
    return (np.ones(10) == control).all()

def create_zeros(control):
    return (np.zeros(10) == control).all()


class TestArray(unittest.TestCase):

    def test_create_arrays(self, flags=enable_pyobj_flags):
        pyfunc = create_array
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,),
                              flags=flags)
        cfunc = cr.entry_point
        control = np.array([1,2,3])
        self.assertTrue(cfunc(control))

    @unittest.expectedFailure
    def test_create_arrays_npm(self):
        self.test_create_arrays(flags=Noflags)

    def test_create_empty_array(self, flags=enable_pyobj_flags):
        pyfunc = create_empty_array
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,),
                              flags=flags)
        cfunc = cr.entry_point
        control = np.array([])
        self.assertTrue(cfunc(control))

    @unittest.expectedFailure
    def test_create_empty_array_npm(self):
        self.test_create_empty_array(flags=Noflags)

    def test_create_arange(self, flags=enable_pyobj_flags):
        pyfunc = create_arange
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,),
                              flags=flags)
        cfunc = cr.entry_point
        control = np.arange(10)
        self.assertTrue(cfunc(control))
        
    @unittest.expectedFailure
    def test_create_arange_npm(self):
        self.test_create_arange(flags=Noflags)

    def test_create_empty(self, flags=enable_pyobj_flags):
        pyfunc = create_empty
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,),
                              flags=flags)

        cfunc = cr.entry_point
        control = np.empty(10)
        self.assertTrue(cfunc(control))

    @unittest.expectedFailure
    def test_create_empty_npm(self):
        self.test_create_empty(flags=Noflags)

    def test_create_ones(self, flags=enable_pyobj_flags):
        pyfunc = create_ones
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,),
                              flags=flags)
        cfunc = cr.entry_point
        control = np.ones(10)
        self.assertTrue(cfunc(control))

    @unittest.expectedFailure
    def test_create_ones_npm(self):
        self.test_create_ones(flags=Noflags)

    def test_create_zeros(self, flags=enable_pyobj_flags):
        pyfunc = create_zeros
        arraytype = types.Array(types.int32, 1, 'C')
        cr = compile_isolated(pyfunc, (arraytype,),
                              flags=flags)
        cfunc = cr.entry_point
        control = np.zeros(10)
        self.assertTrue(cfunc(control))

    @unittest.expectedFailure
    def test_create_zeros_npm(self):
        self.test_create_zeros(flags=Noflags)


if __name__ == '__main__':
    unittest.main()

