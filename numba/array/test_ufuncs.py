from __future__ import print_function, absolute_import
import numba.unittest_support as unittest
import numba.array as numbarray
import numpy as np
from numba.config import PYVERSION
from math import pi
from functools import wraps 
#from numba.types import integer_domain, real_domain
use_python = False

class TestUFuncs(unittest.TestCase):
  
    # todo test different dtypes
    def unary_ufunc_test(self, func_name, data='zeros', scalar=1,
                         size=10, types=[], debug=False):

        if data not in ['zeros', 'ones', 'arange']:
            raise ValueError('{0} is not a valid value for data '
                             'argument'.format(data))

        numba_func = getattr(numbarray, func_name)
        numpy_func = getattr(np, func_name)

        #size = 10
        if not types:
            # 'f2' datatype fails 
            dts = [ 'i4', 'i8', 'u4', 'u8', 'f4', 'f8']
            # todo incorporate with numba/types.py
            #dts = integer_domain | real_domain 
        else:
            dts = types
        for dt in dts:

            a = getattr(numbarray, data)(size, dtype=dt)
            b = getattr(np, data)(size, dtype=dt)

            result = numba_func(a)
            result = result.eval()
            expected = numpy_func(b)

            if debug:
                # todo -- numba_func.__name__ prints 'unary op' rather than
                # the correct name
                print()
                print('testing ' + numpy_func.__name__  + ' with data type ' + dt)
                print('\n' + numpy_func.__name__ + ' test')
                #print numba_func
                print('result = ', result)
                print('expected = ', expected)

            self.assertTrue(np.allclose(result, expected))

    def binary_ufunc_test(self, func_name, data='zeros', m=3, n=3,
                          ascalar=1, bscalar=1, agiven=[], bgiven=[], size=10,
                          types=[], debug=False):

        if data not in ['m_zeros', 'zeros', 'm_ones', 'ones', 'arange', 'given']:
            raise ValueError('{0} is not a valid value for data '
                             'argument'.format(data))

        if data in ['m_zeros', 'm_ones']:
            data = data[2:]
            size = (m, n)

        numba_func = getattr(numbarray, func_name)
        numpy_func = getattr(np, func_name)

        if not types:
            dts = ['i4', 'i8', 'u4', 'u8', 'f4', 'f8']
        else:
            dts = types
        for dt in dts:

            if data == 'given':
                a = numbarray.array(agiven, dtype=dt)
                b = numbarray.array(bgiven, dtype=dt)
                c = numbarray.array(agiven, dtype=dt)
                d = numbarray.array(bgiven, dtype=dt)
            else:
                a = getattr(numbarray, data)(size, dtype=dt)
                b = getattr(numbarray, data)(size, dtype=dt)
                c = getattr(np, data)(size, dtype=dt)
                d = getattr(np, data)(size, dtype=dt)
            
            result = numba_func(a, b)
            result = result.eval()
            expected = numpy_func(c, d)

            if debug:
                print()
                print('testing ' + numpy_func.__name__  + ' with data type ' + dt)
                # todo -- numba_func.__name__ prints 'unary op' rather than
                # the correct name
                print('\n' + numpy_func.__name__ + ' test')
                #print numba_func
                print('result = ', result)
                print(type(result))
                print('expected = ', expected)
                print(type(expected))

            self.assertTrue(np.allclose(result, expected))

    #####################
    # unary ufunc tests #
    #####################

    def test_unary_sin_ufunc(self):
        self.unary_ufunc_test('sin', 'zeros')

    def test_unary_cos_ufunc(self):
        self.unary_ufunc_test('cos', 'zeros')

    def test_unary_tan_ufunc(self):
        
        self.unary_ufunc_test('tan', 'ones', numbarray.pi / 4)
        #self.unary_ufunc_test(numbarray.tan, np.tan, 'ones', numbarray.pi / 4,
        #                      types=['f4'], debug=True)

    def test_unary_arcsin_ufunc(self):
        self.unary_ufunc_test('arcsin', 'zeros')

    def test_unary_arccos_ufunc(self):
        self.unary_ufunc_test('arccos', 'ones')
    
    def test_unary_arctan_ufunc(self):
        self.unary_ufunc_test('arctan', 'ones')

    def test_unary_degrees_ufunc(self):
        self.unary_ufunc_test('degrees', 'arange', numbarray.pi / 6) 
 
    # same as the degrees ufunc
    def test_unary_rad2deg_ufunc(self):
        self.unary_ufunc_test('rad2deg', 'arange', numbarray.pi / 6) 
       
    # same as the radians ufunc   
    def test_unary_deg2rad_ufunc(self):
        self.unary_ufunc_test('deg2rad', 'arange', 30) 

    def test_unary_radians_ufunc(self):
        self.unary_ufunc_test('radians', 'arange', 30) 

    # todo -- complex types not implemented yet 
    # should test hyperbolics with complex datatypes
    # a = numbarray.zeros(size) * numbarray.pi * 1j / 2
    # b = np.arange(size) * np.pi * 1j / 2
    
    def test_unary_sinh_ufunc(self):
        self.unary_ufunc_test('sinh', 'zeros') 
        
    def test_unary_cosh_ufunc(self):
        self.unary_ufunc_test('cosh', 'zeros') 
    
    def test_unary_tanh_ufunc(self):
        #self.unary_ufunc_test(numbarray.tanh, np.tanh, 'ones', numbarray.pi / 4)
        self.unary_ufunc_test('tanh', 'ones')

    def test_unary_arcsinh_ufunc(self):
        self.unary_ufunc_test('arcsinh', 'ones', numbarray.e)

    def test_unary_arccosh_ufunc(self):
        self.unary_ufunc_test('arccosh', 'ones', numbarray.e)

    def test_unary_arctanh_ufunc(self):
        self.unary_ufunc_test('arctanh', 'ones', numbarray.e)

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
        self.unary_ufunc_test('floor', 'ones', numbarray.e)
 
    def test_unary_ceil_ufunc(self):
        self.unary_ufunc_test('ceil', 'ones', numbarray.e)

    def test_unary_trunc_ufunc(self):
        self.unary_ufunc_test('trunc', 'arange', numbarray.e)
    
    def test_unary_rint_ufunc(self):
        self.unary_ufunc_test('rint', 'arange', numbarray.e)
    
    def test_unary_exp_ufunc(self):
        self.unary_ufunc_test('exp', 'arange', numbarray.e)

    def test_unary_exp2_ufunc(self):
        self.unary_ufunc_test('exp2', 'arange', numbarray.e)
 
    def test_unary_expm1_ufunc(self):
        self.unary_ufunc_test('expm1', 'arange', numbarray.e)

    def test_unary_log_ufunc(self):
        self.unary_ufunc_test('log', 'arange')
        # these dtypes work
        #self.unary_ufunc_test(numbarray.log, np.log, 'arange',
        #                      types=['i4', 'i8', 'u4', 'u8', 'f8'])

    def test_unary_log2_ufunc(self):
        self.unary_ufunc_test('log2', 'arange')
        # these dytpes work
        #self.unary_ufunc_test(numbarray.log2, np.log2, 'arange',
        #                      types=['i4', 'i8', 'u4', 'u8', 'f8'])

    def test_unary_log10_ufunc(self):
        # these dtypes work, probably similar for other failures
        #self.unary_ufunc_test(numbarray.log10, np.log10, 'arange',
        #                      types=['i4', 'i8', 'u4', 'u8', 'f8'])
        self.unary_ufunc_test('log10', 'arange')

    def test_unary_log1p_ufunc(self):
        self.unary_ufunc_test('log1p', 'arange')
        # these dtypes work, probably similar for other failures
        #self.unary_ufunc_test(numbarray.log10, np.log10, 'arange',
        #                      types=['i4', 'i8', 'u4', 'u8', 'f8'])

# todo -- frexp test fails with the following error
# File "/home/scott/continuum/numba/numba/typeinfer.py", line 114, in propagate
#    raise TypingError("Internal error:\n%s" % e, constrain.loc)
# TypingError: Internal error:
#    Attribute 'frexp' of Module(<module 'math' from
#    '/home/scott/anaconda/lib/python2.7/lib-dynload/math.so'>) is not typed
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
        self.unary_ufunc_test('negative', 'arange',
                              types=['i1', 'i2', 'i4', 'i4', 'f4', 'f8'])
    
    def test_unary_sign_ufunc(self):
        self.unary_ufunc_test('sign', 'ones') 

    def test_unary_sqrt_ufunc(self):
        self.unary_ufunc_test('sqrt', 'arange')
        # these tests work
        #self.unary_ufunc_test(numbarray.sqrt, np.sqrt, 'arange',
        #                      types=['i4', 'i8', 'u4', 'u8', 'f8']) 

    def test_unary_fabs_ufunc(self):
        self.unary_ufunc_test('fabs', 'arange', scalar=-1)

    def test_unary_absolute_ufunc(self):
        self.unary_ufunc_test('absolute', 'arange')

    def test_unary_conj_ufunc(self):
        self.unary_ufunc_test('conj', 'arange')

    def test_unary_square_ufunc(self):
        self.unary_ufunc_test('square', 'arange')

    def test_unary_reciprocal_ufunc(self):
        self.unary_ufunc_test('reciprocal', 'arange')

    def test_unary_invert_ufunc(self):
        self.unary_ufunc_test('invert', 'arange')

    def test_unary_logical_not_ufunc(self):
        self.unary_ufunc_test('logical_not', 'arange')

    def test_unary_isinfinite_ufunc(self):
        self.unary_ufunc_test('isinfinite', 'arange')

    def test_unary_isinf_ufunc(self):
        self.unary_ufunc_test('isinf', 'arange')

    def test_unary_isnan_ufunc(self):
        self.unary_ufunc_test('isnan', 'arange')

    def test_unary_signbit_ufunc(self):
        self.unary_ufunc_test('signbit', 'arange')

    def test_unary_modf_ufunc(self):
        self.unary_ufunc_test('modf', 'arange')

    def test_unary_frexp_ufunc(self):
        self.unary_ufunc_test('frexp', 'arange')

    ######################
    # binary ufunc tests #
    ######################

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

    def test_binary_subtract_ufunc(self):
        self.binary_ufunc_test('subtract', 'arange') 

    def test_binary_multiply_ufunc(self):
        self.binary_ufunc_test('multiply', 'arange')

    def test_binary_divide_ufunc(self):
        self.binary_ufunc_test('divide', 'arange')

    def test_binary_true_divide_ufunc(self):
        self.binary_ufunc_test('true_divide', 'arange')

    def test_binary_floor_divide_ufunc(self):
        self.binary_ufunc_test('floor_divide', 'arange')

    def test_binary_power_ufunc(self):
        self.binary_ufunc_test('power', 'arange')

    def test_binary_hypot_ufunc(self):
        self.binary_ufunc_test('hypot', 'm_ones', ascalar=3, bscalar=4)
    
    def test_binary_arctan2_ufunc(self):
        # fails with 'il' dtype
        self.binary_ufunc_test('arctan2', 'given',
                               agiven=[-1, +1, +1, -1, 0],
                               bgiven=[-1, -1, +1, +1, 0],
                               types=['i2', 'i4', 'f4', 'f8'],
                               ascalar=180 / numbarray.pi,
                               bscalar=180 / numbarray.pi)

    def test_binary_logaddexp_ufunc(self):
        self.binary_ufunc_test('logaddexp', 'arange')

    def test_binary_logaddexp2_ufunc(self):
        self.binary_ufunc_test('logaddexp2', 'arange')

    def test_binary_remainder_ufunc(self):
        self.binary_ufunc_test('remainder', 'arange')

    def test_binary_mod_ufunc(self):
        self.binary_ufunc_test('mod', 'arange')
       
    def test_binary_fmod_ufunc(self):
        self.binary_ufunc_test('fmod', 'arange')

    def test_binary_bitwise_and_ufunc(self):
        self.binary_ufunc_test('bitwise_and', 'arange')
       
    def test_binary_bitwise_or_ufunc(self):
        self.binary_ufunc_test('bitwise_or', 'arange')
       
    def test_binary_bitwise_xor_ufunc(self):
        self.binary_ufunc_test('bitwise_xor', 'arange')
       
    def test_binary_left_shift_ufunc(self):
        self.binary_ufunc_test('left_shift', 'arange')
       
    def test_binary_right_shift_ufunc(self):
        self.binary_ufunc_test('right_shift', 'arange')
       
    def test_binary_greater_ufunc(self):
        self.binary_ufunc_test('greater', 'arange')
       
    def test_binary_greater_equal_ufunc(self):
        self.binary_ufunc_test('greater_equal', 'arange')
       
    def test_binary_less_ufunc(self):
        self.binary_ufunc_test('less', 'arange')
       
    def test_binary_less_equal_ufunc(self):
        self.binary_ufunc_test('less_equal', 'arange')
       
    def test_binary_not_equal_ufunc(self):
        self.binary_ufunc_test('not_equal', 'arange')
       
    def test_binary_equal_ufunc(self):
        self.binary_ufunc_test('equal', 'arange')
       
    def test_binary_logical_and_ufunc(self):
        self.binary_ufunc_test('logical_and', 'arange')
       
    def test_binary_logical_or_ufunc(self):
        self.binary_ufunc_test('logical_or', 'arange')
       
    def test_binary_logical_xor_ufunc(self):
        self.binary_ufunc_test('logical_xor', 'arange')
       
    def test_binary_maximum_ufunc(self):
        self.binary_ufunc_test('maximum', 'arange')
       
    def test_binary_minimum_ufunc(self):
        self.binary_ufunc_test('minimum', 'arange')
       
    def test_binary_fmax_ufunc(self):
        self.binary_ufunc_test('fmax', 'arange')
       
    def test_binary_fmin_ufunc(self):
        self.binary_ufunc_test('fmin', 'arange')
       
    def test_binary_copysign_ufunc(self):
        self.binary_ufunc_test('copysign', 'arange')
       
    def test_binary_nextafter_ufunc(self):
        self.binary_ufunc_test('nextafter', 'arange')
       
    def test_binary_ldexp_ufunc(self):
        self.binary_ufunc_test('ldexp', 'arange')
       
    def test_binary_fmod_ufunc(self):
        self.binary_ufunc_test('fmod', 'arange')
       
if __name__ == '__main__':
    unittest.main(verbosity=2)
