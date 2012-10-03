#! /usr/bin/env python
# ______________________________________________________________________
'''test_type_inference

Test type inference.
'''
# ______________________________________________________________________

from numba.minivect import minitypes
from numba import *
from numba import _numba_types as numba_types
from numba import ast_type_inference
from numba import decorators, functions

import unittest

import numpy

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

    # 3D
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

    # 3D
    #j = a[n, None, 0]
    k = a[None, 0, numpy.newaxis]
    l = a[0, n, numpy.newaxis]

# ______________________________________________________________________

def infer(func, argtypes):
    sig = minitypes.FunctionType(return_type=None, args=argtypes)
    ast = functions._get_ast(func)

    sig, symtab, ast = ast_type_inference.run_pipeline(
                                decorators.context, func, ast, sig)
    return sig, symtab

class TestTypeInference(unittest.TestCase):
    def test_simple_func(self):
         self.assertEqual(simple_func(-1.), 42.)
         self.assertEqual(simple_func(1.), 22.)

    def test_simple_for(self):
         self.assertEqual(for_loop(0, 10, 1), 45)

    def test_type_infer_simple_func(self):
        sig, symtab = infer(_simple_func, [double])
        self.assertEqual(sig.return_type, double)

    def test_type_infer_for_loop(self):
        sig, symtab = infer(_for_loop, [int_, int_, int_])
        self.assertEqual(symtab['acc'].type, int_)
        self.assertEqual(symtab['value'].type, Py_ssize_t)
        self.assertEqual(sig.return_type, int_)

    def test_type_infer_arange(self):
        sig, symtab = infer(arange, [])
        self.assertEqual(symtab['a'].type, int64[:])
        self.assertEqual(symtab['b'].type, double[:])

    def test_empty_like(self):
        sig, symtab = infer(empty_like, [double[:]])
        self.assertEqual(symtab['b'].type, double[:])
        self.assertEqual(symtab['c'].type, int32[:])
        self.assertEqual(symtab['d'].type, double[:])

    def test_slicing(self):
        sig, symtab = infer(slicing, [double[:]])
        self.assertEqual(symtab['n'].type, numba_types.NewAxisType())

        self.assertEqual(symtab['b'].type, double)
        self.assertEqual(symtab['c'].type, double)
        self.assertEqual(symtab['d'].type, double[:])
        self.assertEqual(symtab['e'].type, double[:])
        self.assertEqual(symtab['f'].type, double[:, :])
        self.assertEqual(symtab['g'].type, double[:, :])
        self.assertEqual(symtab['h'].type, double[:, :])
        self.assertEqual(symtab['i'].type, double[:, :])
        self.assertEqual(symtab['j'].type, double[:, :, :])
        self.assertEqual(symtab['k'].type, double[:, :, :])
        self.assertEqual(symtab['l'].type, double[:, :, :])

    def test_none_newaxis(self):
        sig, symtab = infer(none_newaxis, [double[:]])
        self.assertEqual(symtab['f'].type, double[:, :])
        #self.assertEqual(symtab['g'].type, double[:, :])
        self.assertEqual(symtab['h'].type, double[:, :])
        #self.assertEqual(symtab['i'].type, double[:, :])
        #self.assertEqual(symtab['j'].type, double[:, :, :])
        self.assertEqual(symtab['k'].type, double[:, :, :])
        self.assertEqual(symtab['l'].type, double[:, :, :])


# ______________________________________________________________________

if __name__ == "__main__":
#    TestTypeInference('test_slicing').debug()
    unittest.main()
