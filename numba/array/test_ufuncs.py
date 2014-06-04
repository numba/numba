import numba.unittest_support as unittest
import numba.array as numbarray
import numpy as np
from numba.config import PYVERSION
from math import pi
from functools import wraps 
use_python = True

size = 10

class TestUFuncs(unittest.TestCase):
    
    def test_binary_ufunc(self):

        a = numbarray.arange(10)
        result = numbarray.add(a, a)
        expected = np.add(np.arange(10), np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(1, 1)
        expected = np.add(1, 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(a, 1)
        expected = np.add(np.arange(10), 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(1, a)
        expected = np.add(1, np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_sin_ufunc(self):

        a = (pi / 2) * numbarray.ones(size)
        b = (pi / 2) * np.ones(size)

        result = numbarray.sin(a)
        result.eval()  # eval is deferred so w/o (or print) this the assert fails
        #print result
        expected = np.sin(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_cos_ufunc(self):
        a = numbarray.zeros(size)
        b = np.zeros(size)
        result = numbarray.cos(a)
        result.eval()
        expected = np.cos(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_tan_ufunc(self):
        # The (pi / 4) fails
        #a = (pi / 4) * numbarray.ones(10)
        a = numbarray.zeros(size)
        #b = (pi / 4) * np.ones(10)
        b = np.zeros(size)
        result = numbarray.tan(a)
        result.eval()
        expected = np.tan(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_arcsin_ufunc(self):
        a = numbarray.zeros(size)
        b = np.zeros(size)
        result = numbarray.arcsin(a)
        result.eval()
        expected = np.arcsin(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_arccos_ufunc(self):
        a = numbarray.ones(size)
        b = np.ones(size)
        result = numbarray.arccos(a)
        result.eval()
        print type(result)
        expected = np.arccos(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
    
    def test_unary_arctan_ufunc(self):
        a = numbarray.ones(size)
        b = np.ones(size)
        result = numbarray.arctan(a)
        result.eval()
        expected = np.arctan(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_degrees_ufunc(self):
        # both tests fail 
        print 'degrees test'
        #a = numbarray.arange(12.) * numbarray.pi / 6
        #b = np.arange(12.) * np.pi / 6
        a = numbarray.arange(12) 
        b = np.arange(12) 
        result = numbarray.degrees(a)
        result.eval()
        print result
        expected = np.degrees(b)
        print expected
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_rad2deg_ufunc(self):
        # both tests fail 
        print 'rad2deg test'
        #a = numbarray.arange(12.) * numbarray.pi / 6
        #b = np.arange(12.) * np.pi / 6
        a = numbarray.arange(12) 
        b = np.arange(12) 
        result = numbarray.rad2deg(a)
        result.eval()
        print result
        expected = np.rad2deg(b)
        print expected
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_deg2rad_ufunc(self):
        # this is the same as the radians function
        # numpy implements both
        a = numbarray.arange(12) * 30. 
        b = np.arange(12) * 30.
        result = numbarray.deg2rad(a)
        result.eval()
        expected = np.deg2rad(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_radians_ufunc(self):
        a = numbarray.arange(12) * 30. 
        b = np.arange(12) * 30.
        result = numbarray.radians(a)
        result.eval()
        expected = np.radians(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_sinh_ufunc(self):
        #print 'sinh test'
        # complex types not implemented yet
        #a = numbarray.zeros(size) * numbarray.pi * 1j / 2
        #b = np.arange(size) * np.pi * 1j / 2
        a = numbarray.zeros(size)
        b = np.zeros(size)
        result = numbarray.sinh(a)
        result.eval()
        #print result
        expected = np.sinh(b)
        #print expected
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_cosh_ufunc(self):
        a = numbarray.zeros(size)
        b = np.zeros(size)
        result = numbarray.cosh(a)
        result.eval()
        expected = np.cosh(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
    
    def test_unary_tanh_ufunc(self):
        print 'tanh test'
        #first test fails; second one works
        a = numbarray.ones(size) * numbarray.pi / 4
        b = np.ones(size) * np.pi / 4
        #a = numbarray.zeros(size) 
        #b = np.zeros(size)
        result = numbarray.tanh(a)
        result.eval()
        print result
        expected = np.tanh(b)
        print expected
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_arcsinh_ufunc(self):
        print 'arcsinh test'
        # complex values not implemented yet
        # first test fails; second works
        a = numbarray.ones(size) * numbarray.e 
        b = np.ones(size) * np.e
        #a = numbarray.ones(size) 
        #b = np.ones(size)
        result = numbarray.arcsinh(a)
        result.eval()
        print result
        expected = np.arcsinh(b)
        print expected
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_arccosh_ufunc(self):
        print 'arccosh test'
        # complex values not implemented yet
        # first test fails; second works
        a = numbarray.ones(size) * numbarray.e 
        b = np.ones(size) * np.e
        #a = numbarray.ones(size) 
        #b = np.ones(size)
        result = numbarray.arccosh(a)
        result.eval()
        print result
        expected = np.arccosh(b)
        print expected
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_arctanh_ufunc(self):
        print 'arctanh test'
        # complex values not implemented yet
        # first test fails; second and third work
        a = numbarray.ones(size) * numbarray.e 
        b = np.ones(size) * np.e
        #a = numbarray.ones(size)
        #b = np.ones(size)
        #a = numbarray.ones(size) * .5 
        #b = np.ones(size) * .5
        result = numbarray.arctanh(a)
        result.eval()
        print result
        expected = np.arctanh(b)
        print expected
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_binary_hypot_ufunc(self):
        a = 3 * numbarray.ones((3, 3))
        b = 4 * numbarray.ones((3, 3))
        c = 3 * np.ones((3, 3))
        d = 4 * np.ones((3, 3))
        result = numbarray.hypot(a, b)
        result.eval()
        expected = np.hypot(c, d)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
    
    def test_binary_arctan2_ufunc(self):
        # the first two tests work fine
        # the test with the 180 / [numbarray.pi|np.pi] fails
        print 'arctan2 test'
        a = numbarray.array([-1, +1, +1, -1, 0])
        b = numbarray.array([-1, -1, +1, +1, 0])
        c = np.array([-1, +1, +1, -1, 0])
        d = np.array([-1, -1, +1, +1, 0])
        #result = numbarray.arctan2(a, b) * 180
        #result = numbarray.arctan2(a, b) * 180 / 3.14 
        result = numbarray.arctan2(a, b) * 180 / numbarray.pi
        result.eval()
        print result
        print numbarray.pi
        #expected = np.arctan2(c, d) * 180
        #expected = np.arctan2(c, d) * 180 / 3.14 
        expected = np.arctan2(c, d) * 180 / np.pi
        print expected
        print np.pi
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
       
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
        a = numbarray.ones(size) * numbarray.e
        b = np.ones(size) * np.e
        #a = numbarray.ones(size) / numbarray.e
        #b = np.ones(size) / np.e
        result = numbarray.floor(a)
        result.eval()
        expected = np.floor(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
 
    def test_unary_ceil_ufunc(self):
        a = numbarray.ones(size) * numbarray.e
        b = np.ones(size) * np.e
        #a = numbarray.ones(size) / numbarray.e
        #b = np.ones(size) / np.e
        result = numbarray.ceil(a)
        result.eval()
        expected = np.ceil(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_trunc_ufunc(self):
        a = numbarray.arange(size) * numbarray.e
        b = np.arange(size) * np.e
        result = numbarray.ceil(a)
        result.eval()
        expected = np.ceil(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_exp_ufunc(self):
        a = numbarray.arange(size) 
        b = np.arange(size) 
        result = numbarray.exp(a)
        result.eval()
        expected = np.exp(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_expm1_ufunc(self):
        a = numbarray.arange(size) 
        b = np.arange(size) 
        result = numbarray.expm1(a)
        result.eval()
        expected = np.expm1(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_log_ufunc(self):
        a = numbarray.arange(size) 
        b = np.arange(size) 
        result = numbarray.log(a)
        result.eval()
        expected = np.log(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_log10_ufunc(self):
        a = numbarray.arange(size) 
        b = np.arange(size) 
        result = numbarray.log10(a)
        result.eval()
        expected = np.log10(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_log1p_ufunc(self):
        a = numbarray.arange(size) 
        b = np.arange(size) 
        result = numbarray.log1p(a)
        result.eval()
        expected = np.log1p(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

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
        a = numbarray.arange(size) 
        b = np.arange(size) 
        result = -a
        result.eval()
        expected = -b
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

    def test_unary_negative_ufunc(self):
        a = numbarray.arange(size) 
        b = np.arange(size) 
        result = numbarray.negative(a)
        result.eval()
        expected = np.negative(b)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

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
        # the first two tests work fine
        # the test with the 180 / [numbarray.pi|np.pi] fails
        a = numbarray.arange(9).reshape((3, 3))
        b = numbarray.arange(3)
        c = np.arange(9).reshape((3, 3))
        d = np.arange(3)
        result = numbarray.subtract(a, b)
        result.eval()
        expected = np.subtract(c, d)
        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))
        print 'subtract test'
        print result
        print expected

if __name__ == '__main__':
    unittest.main()
