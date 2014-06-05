import numba.unittest_support as unittest
import numba.array as numbarray
import numpy as np
from numba.config import PYVERSION
from math import pi
from functools import wraps 
use_python = False

class TestUFuncs(unittest.TestCase):
  
    # todo test different dtypes
    def unary_ufunc_test(self, numba_func, numpy_func, data='zeros', scalar=1, size=10, types=[], debug=False):
        #size = 10
        if not types:
            dts = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8']
        else:
            dts = types
        print
        for dt in dts:
            print 'testing ' + numpy_func.__name__  + ' with data type ' + dt
            if data == 'zeros':
                a = numbarray.zeros(size, dtype=dt)
                b = np.zeros(size, dtype=dt)
            elif data == 'ones':
                a = scalar * numbarray.ones(size, dtype=dt)
                b = scalar * np.ones(size, dtype=dt)
            elif data == 'arange':
                a = scalar * numbarray.arange(size, dtype=dt)
                b = scalar * np.arange(size, dtype=dt)
            result = numba_func(a)
            result.eval()
            expected = numpy_func(b)
            if debug:
                # todo -- numba_func.__name__ prints 'unary op' rather than the correct name
                print '\n' + numpy_func.__name__ + ' test'
                #print numba_func
                print 'result = ', result
                print 'expected = ', expected
            self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def binary_ufunc_test(self, numba_func, numpy_func, data='zeros', m=3, n=3, ascalar=1, bscalar=1, agiven=[], bgiven=[], size=10, types=[], debug=False):
        if not types:
            dts = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8']
        else:
            dts = types
        print 
        for dt in dts:
            print 'testing ' + numpy_func.__name__  + ' with data type ' + dt
            if data == 'm_zeros':
                a = ascalar * numbarray.zeros((m, n), dtype=dt)
                b = bscalar * numbarray.zeros((m, n), dtype=dt)
                c = ascalar * np.zeros((m, n), dtype=dt)
                d = bscalar * np.zeros((m, n), dtype=dt)
            elif data == 'zeros':
                a = ascalar * numbarray.zeros(size, dtype=dt)
                b = bscalar * numbarray.zeros(size, dtype=dt)
                c = ascalar * np.zeros(size, dtype=dt)
                d = bscalar * np.zeros(size, dtype=dt)
            elif data == 'm_ones':
                a = ascalar * numbarray.ones((m, n), dtype=dt)
                b = bscalar * numbarray.ones((m, n), dtype=dt)
                c = ascalar * np.ones((m, n), dtype=dt)
                d = bscalar * np.ones((m, n), dtype=dt)
            elif data == 'ones':
                a = ascalar * numbarray.ones(size, dtype=dt)
                b = bscalar * numbarray.ones(size, dtype=dt)
                c = ascalar * np.ones(size, dtype=dt)
                d = bscalar * np.ones(size, dtype=dt)
            elif data == 'arange':
                a = ascalar * numbarray.arange(size, dtype=dt)
                b = bscalar * numbarray.arange(size, dtype=dt)
                c = ascalar * np.arange(size, dtype=dt)
                d = bscalar * np.arange(size, dtype=dt)
            elif data == 'given':
                a = ascalar * numbarray.array(agiven, dtype=dt)
                b = bscalar * numbarray.array(bgiven, dtype=dt)
                c = ascalar * np.array(agiven, dtype=dt)
                d = bscalar * np.array(bgiven, dtype=dt)

            
            result = numba_func(a, b)
            result.eval()
            expected = numpy_func(c, d)
            if debug:
                # todo -- numba_func.__name__ prints 'unary op' rather than the correct name
                print '\n' + numpy_func.__name__ + ' test'
                #print numba_func
                print 'result = ', result
                print type(result)
                print 'expected = ', expected
                print type(expected)
            self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    #  renamed from test_bunary_ufunc
    def test_binary_add_ufunc(self):

        a = numbarray.arange(10)
        result = numbarray.add(a, a)
        expected = np.add(np.arange(10), np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

       # result = numbarray.add(1, 1)
       # expected = np.add(1, 1)

       # self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(a, 1)
        expected = np.add(np.arange(10), 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        resulst = numbarray.add(1, a)
        expected = np.add(1, np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
        
    def test_unary_sin_ufunc(self):
        self.unary_ufunc_test(numbarray.sin, np.sin, 'zeros')

    def test_unary_cos_ufunc(self):
        self.unary_ufunc_test(numbarray.cos, np.cos, 'zeros')

    def test_unary_tan_ufunc(self):
        
        self.unary_ufunc_test(numbarray.tan, np.tan, 'ones', numbarray.pi / 4)
        #self.unary_ufunc_test(numbarray.tan, np.tan, 'ones', numbarray.pi / 4, types=['f4'], debug=True)

    def test_unary_arcsin_ufunc(self):
        self.unary_ufunc_test(numbarray.arcsin, np.arcsin, 'zeros')

    def test_unary_arccos_ufunc(self):
        self.unary_ufunc_test(numbarray.arccos, np.arccos, 'ones')
    
    def test_unary_arctan_ufunc(self):
        self.unary_ufunc_test(numbarray.arctan, np.arctan, 'ones')

    def test_unary_degrees_ufunc(self):
        self.unary_ufunc_test(numbarray.degrees, np.degrees, 'arange', numbarray.pi / 6) 
 
    # same as the degrees ufunc
    def test_unary_rad2deg_ufunc(self):
        self.unary_ufunc_test(numbarray.rad2deg, np.rad2deg, 'arange', numbarray.pi / 6) 
       
    # same as the radians ufunc   
    def test_unary_deg2rad_ufunc(self):
        self.unary_ufunc_test(numbarray.deg2rad, np.deg2rad, 'arange', 30) 

    def test_unary_radians_ufunc(self):
        self.unary_ufunc_test(numbarray.radians, np.radians, 'arange', 30) 

    # todo -- complex types not implemented yet 
    # should test hyperbolics with complex datatypes
    # a = numbarray.zeros(size) * numbarray.pi * 1j / 2
    # b = np.arange(size) * np.pi * 1j / 2
    
    def test_unary_sinh_ufunc(self):
        self.unary_ufunc_test(numbarray.sinh, np.sinh, 'zeros') 
        
    def test_unary_cosh_ufunc(self):
        self.unary_ufunc_test(numbarray.cosh, np.cosh, 'zeros') 
    
    def test_unary_tanh_ufunc(self):
        #self.unary_ufunc_test(numbarray.tanh, np.tanh, 'ones', numbarray.pi / 4)
        self.unary_ufunc_test(numbarray.tanh, np.tanh, 'ones')

    def test_unary_arcsinh_ufunc(self):
        self.unary_ufunc_test(numbarray.arcsinh, np.arcsinh, 'ones', numbarray.e)

    def test_unary_arccosh_ufunc(self):
        self.unary_ufunc_test(numbarray.arccosh, np.arccosh, 'ones', numbarray.e)

    def test_unary_arctanh_ufunc(self):
        self.unary_ufunc_test(numbarray.arctanh, np.arctanh, 'ones', numbarray.e)

    def test_binary_hypot_ufunc(self):
        self.binary_ufunc_test(numbarray.hypot, np.hypot, 'm_ones', ascalar=3, bscalar=4)
    
    def test_binary_arctan2_ufunc(self):
        # fails with 'il' dtype
        self.binary_ufunc_test(numbarray.arctan2, np.arctan2, 'given', agiven=[-1, +1, +1, -1, 0], bgiven=[-1, -1, +1, +1, 0], types=['i2', 'i4', 'f4', 'f8'], ascalar=180 / numbarray.pi, bscalar=180 / numbarray.pi)

       
    def test_div(self):
        a = 180 / numbarray.pi
        b = 180 / np.pi
        self.assertTrue(a == b)

    def test_floor_division(self):
        a = 180 // numbarray.pi
        b = 180 // np.pi
        self.assertTrue(a == b)

    def test_pi(self):
        self.assertTrue(numbarray.pi == np.pi)

    def test_unary_floor_ufunc(self):
        self.unary_ufunc_test(numbarray.floor, np.floor, 'ones', numbarray.e)
 
    def test_unary_ceil_ufunc(self):
        self.unary_ufunc_test(numbarray.ceil, np.ceil, 'ones', numbarray.e)

    def test_unary_trunc_ufunc(self):
        self.unary_ufunc_test(numbarray.trunc, np.trunc, 'arange', numbarray.e)
    
    def test_unary_exp_ufunc(self):
        self.unary_ufunc_test(numbarray.exp, np.exp, 'arange', numbarray.e)

    def test_unary_expm1_ufunc(self):
        self.unary_ufunc_test(numbarray.expm1, np.expm1, 'arange', numbarray.e)

    def test_unary_log_ufunc(self):
        self.unary_ufunc_test(numbarray.log, np.log, 'arange')
        # these dtypes work
        #self.unary_ufunc_test(numbarray.log, np.log, 'arange', types=['i4', 'i8', 'u4', 'u8', 'f8'])

    def test_unary_log2_ufunc(self):
        self.unary_ufunc_test(numbarray.log2, np.log2, 'arange')
        # these dytpes work
        #self.unary_ufunc_test(numbarray.log2, np.log2, 'arange', types=['i4', 'i8', 'u4', 'u8', 'f8'])

    def test_unary_log10_ufunc(self):
        # these dtypes work, probably similar for other failures
        #self.unary_ufunc_test(numbarray.log10, np.log10, 'arange', types=['i4', 'i8', 'u4', 'u8', 'f8'])
        self.unary_ufunc_test(numbarray.log10, np.log10, 'arange')

    def test_unary_log1p_ufunc(self):
        self.unary_ufunc_test(numbarray.log1p, np.log1p, 'arange')
        # these dtypes work, probably similar for other failures
        #self.unary_ufunc_test(numbarray.log10, np.log10, 'arange', types=['i4', 'i8', 'u4', 'u8', 'f8'])

# todo -- frexp test fails with the following error
# File "/home/scott/continuum/numba/numba/typeinfer.py", line 114, in propagate
#    raise TypingError("Internal error:\n%s" % e, constrain.loc)
# TypingError: Internal error:
#    Attribute 'frexp' of Module(<module 'math' from '/home/scott/anaconda/lib/python2.7/lib-dynload/math.so'>) is not typed
#
#    def test_unary_frexp_ufunc(self):
#        a = numbarray.arange(size) 
#        b = np.arange(size) 
#        result1, result2 = numbarray.frexp(a)
#        result1.eval()
#        result2.eval()
#        expected1 = np.frexp(b)
#        expected2 = np.frexp(b)
#        self.assertTrue(np.all(result1.eval(use_python=use_python) == expected1))
#        self.assertTrue(np.all(result2.eval(use_python=use_python) == expected2))
#        print 'frexp test'
#        print result1
#        print expected1
#        print result2
#        print expected2

    def test_unary_neg_ufunc(self):
        a = numbarray.arange(10) 
        b = np.arange(10) 
        result = -a
        result.eval()
        expected = -b
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_negative_ufunc(self):
        #a = numbarray.arange(size) 
        self.unary_ufunc_test(numbarray.negative, np.negative, 'arange', types=['i1', 'i2', 'i4', 'i4', 'f4', 'f8'])

#  pow and power both give the following error 
#  File "/home/scott/anaconda/lib/python2.7/site-packages/llvm/core.py", line 593, in verify
#      raise llvm.LLVMException(errio.getvalue())
#  LLVMException: Intrinsic has incorrect return type!
#  i64 (i64, i32)* @llvm.powi.i64
#  Broken module found, compilation terminated.
#
#    def test_binary_pow_ufunc(self):
#        a = numbarray.arange(size) 
#        b = np.arange(size) 
#        result = a ** 2
#        result.eval()
#        expected = b ** 2
#        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
#        print 'power test'
#        print result
#        print expected
#
#    def test_binary_power2_ufunc(self):
#        a = numbarray.arange(size) 
#        b = np.arange(size) 
#        result = numbarray.power(a, 2) 
#        result.eval()
#        expected = np.power(b, 2)
#        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
#        print 'pow test'
#        print result
#        print expected

    def test_binary_subtract_ufunc(self):
        self.binary_ufunc_test(numbarray.subtract, np.subtract, 'arange') 

if __name__ == '__main__':
    unittest.main()
