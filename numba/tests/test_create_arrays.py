from __future__ import print_function
import unittest
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types, utils
from numba.tests import usecases

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

force_pyobj_flags = Flags()
force_pyobj_flags.set("force_pyobject")


def create_array(control):
    return (np.array([1,2,3]) == control).all()

def create_empty_array(control):
    return (np.array([]) == control).all()

def create_arange(control):
    return (np.arange(10) == control).all()

def create_empty(control):
    return (np.empty(10) == control).all()

def create_ones(control):
    return (np.ones(10) == control).all()

def create_zeros(control):
    return (np.zeros(10) == control).all()


class TestArray(unittest.TestCase):

    def test_create_arrays(self):
        
        arraytype = types.Array(types.int32, 1, 'C')

        pyfunc = create_array
        cr = compile_isolated(pyfunc, (arraytype,))
        cfunc = cr.entry_point
        control = np.array([1,2,3])
        self.assertTrue(cfunc(control))

        pyfunc = create_empty_array
        cr = compile_isolated(pyfunc, (arraytype,))
        cfunc = cr.entry_point
        control = np.array([])
        self.assertTrue(cfunc(control))

        pyfunc = create_arange
        cr = compile_isolated(pyfunc, (arraytype,))
        cfunc = cr.entry_point
        control = np.arange(10)
        self.assertTrue(cfunc(control))
        
        pyfunc = create_empty
        cr = compile_isolated(pyfunc, (arraytype,))
        cfunc = cr.entry_point
        control = np.empty(10)
        self.assertTrue(cfunc(control))

        pyfunc = create_ones
        cr = compile_isolated(pyfunc, (arraytype,))
        cfunc = cr.entry_point
        control = np.ones(10)
        self.assertTrue(cfunc(control))

        pyfunc = create_zeros
        cr = compile_isolated(pyfunc, (arraytype,))
        cfunc = cr.entry_point
        control = np.zeros(10)
        self.assertTrue(cfunc(control))


if __name__ == '__main__':
    unittest.main()

