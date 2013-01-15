#! /usr/bin/env python
# ______________________________________________________________________
'''test_type_inference

Test type inference.
'''
# ______________________________________________________________________

from numba.minivect import minitypes, minierror
from numba import *
from numba import typesystem
from numba import ast_type_inference
from numba import decorators, functions, pipeline

import unittest

import numpy
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)
# ______________________________________________________________________

def _simple_func(arg):
    if arg > 0.:
        result = 22.
    else:
        result = 42.
    return result

simple_func = decorators.autojit(backend='ast')(_simple_func)

def _for_loop(start, stop, inc):
    acc = 0
    for value in range(start, stop, inc):
        acc += value

    print value
    return acc

for_loop = decorators.autojit(backend='ast')(_for_loop)

def arange():
    a = numpy.arange(10)
    b = numpy.arange(10, dtype=numpy.double)
    return a, b

def empty_like(a):
    b = numpy.empty_like(a)
    c = numpy.zeros_like(a, dtype=numpy.int32)
    d = numpy.ones_like(a)

dtype = np.float32

def _empty(N):
    # default dtype
    a1 = numpy.empty(N)
    a2 = numpy.empty((N,))
    a3 = numpy.empty([N])

    # Given dtype
    a4 = numpy.empty(N, dtype)
    a5 = numpy.empty(N, dtype=dtype)
    a6 = numpy.empty(N, np.float32)
    a7 = numpy.empty(N, dtype=np.float32)

    # Test dimensionality
    a8 = np.empty((N, N), dtype=np.int64)

def _empty_arg(N, empty, zeros, ones):
    a1 = empty([N])
    a2 = zeros([N])
    a3 = ones([N])

@autojit
def assert_array_dtype(A, value, empty, zeros, ones):
    if value < 2:
        A = empty([A.shape[0], A.shape[1]], dtype=A.dtype)
    elif value < 4:
        A = zeros([A.shape[0], A.shape[1]], dtype=A.dtype)
    elif value < 6:
        A = ones([A.shape[0], A.shape[1]], dtype=A.dtype)
    else:
        pass

    # 'A' must have a consistent array type here
    return A

def slicing(a):
    n = numpy.newaxis

    # 0D
    b = a[0]
    c = a[9]

    # 1D
    d = a[:]
    e = a[...]

    # 2D
    f = a[n, ...]
    g = a[numpy.newaxis, :]
    h = a[..., numpy.newaxis]
    i = a[:, n]

    j = a[n, numpy.newaxis, 0]
    k = a[numpy.newaxis, 0, n]
    l = a[0, n, n]

def none_newaxis(a):
    n = None

    # 2D
    f = a[None, ...]
    #g = a[n, :]
    h = a[..., None]
    #i = a[:, n]

    #j = a[n, None, 0]
    k = a[None, 0, numpy.newaxis]
    l = a[0, n, numpy.newaxis]

def func_with_signature(a):
    if a > 1:
        return float(a)
    elif a < 5:
        return int(a)
    elif a > 10:
        return object()

    return a + 1j

def arg_rebind(a):
    a = 0
    a = 0.0
    a = "hello"

# ______________________________________________________________________

from numba.tests.cfg.test_cfg_type_infer import infer, functype

class TestTypeInference(unittest.TestCase):
    def test_simple_func(self):
         self.assertEqual(simple_func(-1.), 42.)
         self.assertEqual(simple_func(1.), 22.)

    def test_simple_for(self):
         self.assertEqual(for_loop(0, 10, 1), 45)

    def test_type_infer_simple_func(self):
        sig, symtab = infer(_simple_func, functype(None, [double]))
        self.assertEqual(sig.return_type, double)

    def test_type_infer_for_loop(self):
        sig, symtab = infer(_for_loop, functype(None, [int_, int_, int_]))
        self.assertTrue(symtab['acc'].type.is_int)
        self.assertEqual(symtab['value'].type, Py_ssize_t)
        self.assertEqual(sig.return_type, Py_ssize_t)

    def test_type_infer_arange(self):
        sig, symtab = infer(arange, functype())
        self.assertEqual(symtab['a'].type, int64[:])
        self.assertEqual(symtab['b'].type, double[:])

    def test_empty_like(self):
        sig, symtab = infer(empty_like, functype(None, [double[:]]))
        self.assertEqual(symtab['b'].type, double[:])
        self.assertEqual(symtab['c'].type, int32[:])
        self.assertEqual(symtab['d'].type, double[:])

    def test_empty(self):
        sig, symtab = infer(_empty, functype(None, [int_]))
        for i in range(1, 4):
            self.assertEqual(symtab['a%d' % i].type, double[:])

        for i in range(4, 8):
            self.assertEqual(symtab['a%d' % i].type, float_[:])

        self.assertEqual(symtab['a8'].type, int64[:, :])

    def test_empty_arg(self):
        from numba import typesystem as nt
        empty_t = nt.ModuleAttributeType(module=np, attr='empty')
        zeros_t = nt.ModuleAttributeType(module=np, attr='zeros')
        ones_t = nt.ModuleAttributeType(module=np, attr='ones')

        sig, symtab = infer(_empty_arg, functype(None, [int_, empty_t,
                                                        zeros_t, ones_t]))
        for i in range(1, 4):
            self.assertEqual(symtab['a%d' % i].type, double[:])

    def test_dtype_attribute(self):
        A = np.empty((10, 10), dtype=np.float32)

        A_result = assert_array_dtype(A, 3, np.empty, np.zeros, np.ones)
        assert np.all(A_result == 0)
        A_result = assert_array_dtype(A, 5, np.empty, np.zeros, np.ones)
        assert np.all(A_result == 1)

    def test_slicing(self):
        sig, symtab = infer(slicing, functype(None, [double[:]]))
        self.assertEqual(symtab['n'].type, typesystem.NewAxisType())

        self.assertEqual(symtab['b'].type, double)
        self.assertEqual(symtab['c'].type, double)
        self.assertEqual(symtab['d'].type, double[:])
        self.assertEqual(symtab['e'].type, double[:])
        self.assertEqual(symtab['f'].type, double[:, :])
        self.assertEqual(symtab['g'].type, double[:, :])
        self.assertEqual(symtab['h'].type, double[:, :])
        self.assertEqual(symtab['i'].type, double[:, :])
        self.assertEqual(symtab['j'].type, double[:, :])
        self.assertEqual(symtab['k'].type, double[:, :])
        self.assertEqual(symtab['l'].type, double[:, :])

    def test_none_newaxis(self):
        sig, symtab = infer(none_newaxis, functype(None, [double[:]]))
        self.assertEqual(symtab['f'].type, double[:, :])
        #self.assertEqual(symtab['g'].type, double[:, :])
        self.assertEqual(symtab['h'].type, double[:, :])
        #self.assertEqual(symtab['i'].type, double[:, :])
        #self.assertEqual(symtab['j'].type, double[:, :, :])
        self.assertEqual(symtab['k'].type, double[:, :])
        self.assertEqual(symtab['l'].type, double[:, :])

    def test_return_type(self):
        sig, symtab = infer(func_with_signature, functype(int_, [int_]))
        assert sig == int_(int_)

        sig, symtab = infer(func_with_signature, functype(int_, [float_]))
        assert sig == int_(float_)

        sig, symtab = infer(func_with_signature, functype(float_, [int_]))
        assert sig == float_(int_)

    def test_rebind_arg(self):
        sig, symtab = infer(arg_rebind, functype(int_, [int_]),
                            allow_rebind_args=True)
        assert sig == int_(int_)
        assert symtab['a'].type == c_string_type

#        try:
#            sig, symtab = infer(arg_rebind, functype(int_, [int_]),
#                                allow_rebind_args=False)
#        except minierror.UnpromotableTypeError, e:
#            msg = str(sorted(e.args, key=str))
#            self.assertEqual("[(double, const char *)]", msg)
#        else:
#            raise Exception("Expected an unpromotable type error")

# ______________________________________________________________________

if __name__ == "__main__":
#    TestTypeInference('test_dtype_attribute').debug()
    unittest.main()
