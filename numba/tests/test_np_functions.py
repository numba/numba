# Tests numpy methods of <class 'function'>
from __future__ import print_function, absolute_import, division

import itertools
import math
import sys

import numpy as np

from numba import unittest_support as unittest
from numba.compiler import compile_isolated, Flags, utils
from numba import types
from .support import TestCase, CompilationCache

no_pyobj_flags = Flags()
no_pyobj_flags.set("nrt")
no_pyobj_flags = [no_pyobj_flags]

def sinc(x):
    return np.sinc(x)

def angle(x, deg):
    return np.angle(x, deg)

class TestNPFunctions(TestCase):
    """
    Contains tests and test helpers for numpy methods the are of type
    "class< 'function' >.
    """

    def setUp(self):
        self.ccache = CompilationCache()

    def run_unary_real(self, pyfunc, x_types, x_values,
        flags=no_pyobj_flags, prec='exact',
        func_extra_types=None, func_extra_args=None,
        ignore_sign_on_zero=False, abs_tol=None, **kwargs):
        """
        Runs tests for a unary function operating in the numerical real space.

        Parameters
        ----------
        pyfunc : a python function definition holding that calls the numpy
                 functions to be tested.
        x_types: the types of the values being tested, see numba.types
        x_values: the numerical values of the values to be tested
        flags: flags to pass to the CompilationCache::ccache::compile function
        func_extra_types: the types of additional arguments to the numpy
                          function
        func_extra_args:  additional arguments to the numpy function
        ignore_sign_on_zero: boolean as to whether to allow zero values
        with incorrect signs to be considered equal
        prec: the required precision match, see assertPreciseEqual

        Notes:
        ------
        x_types and x_values must have the same length

        """
        for f in flags:
            for tx, vx in zip(x_types, x_values):
                if func_extra_args == None:
                    cr = self.ccache.compile(pyfunc, (tx,), flags=f)
                    cfunc = cr.entry_point
                    got = cfunc(vx)
                    expected = pyfunc(vx)
                    actual_prec = 'single' if tx is types.float32 else prec
                    msg = 'for input %r with prec %r' % (vx, prec)
                    self.assertPreciseEqual(got, expected,
                                            prec=actual_prec, msg=msg,
                                            ignore_sign_on_zero=
                                            ignore_sign_on_zero,
                                            abs_tol=abs_tol, **kwargs)
                else:
                    for xtype, xargs in zip(func_extra_types, func_extra_args):
                        cr = self.ccache.compile(pyfunc, (tx,xtype), flags=f)
                        cfunc = cr.entry_point
                        got = cfunc(vx, xargs)
                        expected = pyfunc(vx, xargs)
                        actual_prec = 'single' if tx is types.float32 else prec
                        msg = 'for input %r with prec %r' % (vx, prec)
                        self.assertPreciseEqual(got, expected,
                                                prec=actual_prec,
                                                msg=msg,
                                                ignore_sign_on_zero=
                                                ignore_sign_on_zero,
                                                abs_tol=abs_tol, **kwargs)

    def run_unary_complex(self, pyfunc, x_types, x_values, ulps=1,
                  func_extra_types=None, func_extra_args=None,
                  ignore_sign_on_zero=False, abs_tol=None,
                  flags=no_pyobj_flags):
        """
        Runs tests for a unary function operating in the numerical complex
        space.

        Parameters
        ----------
        pyfunc : a python function definition holding that calls the numpy
                 functions to be tested.
        x_types: the types of the values being tested, see numba.types
        x_values: the numerical values of the values to be tested
        ulps: the number of ulps in error considered acceptable
        flags: flags to pass to the CompilationCache::ccache::compile function
        func_extra_types: the types of additional arguments to the numpy
                          function
        func_extra_args:  additional arguments to the numpy function
        ignore_sign_on_zero: boolean as to whether to allow zero values
        with incorrect signs to be considered equal
        prec: the required precision match, see assertPreciseEqual

        Notes:
        ------
        x_types and x_values must have the same length
        """
        for f in flags:
            for tx in x_types:
                if(func_extra_args == None):
                    cr = self.ccache.compile(pyfunc, (tx,), flags=f)
                    cfunc = cr.entry_point
                    prec = 'single' if tx in (types.float32, types.complex64) \
                        else 'double'
                    for vx in x_values:
                        try:
                            expected = pyfunc(vx)
                        except ValueError as e:
                            self.assertIn("math domain error", str(e))
                            continue
                        got = cfunc(vx)
                        msg = 'for input %r with prec %r' % (vx, prec)
                        self.assertPreciseEqual(got, expected, prec=prec,
                                                ulps=ulps, msg=msg,
                                                ignore_sign_on_zero=
                                                ignore_sign_on_zero,
                                                abs_tol=abs_tol)
                else:
                    for xtype, xargs in zip(func_extra_types, func_extra_args):
                        cr = self.ccache.compile(pyfunc, (tx,xtype), flags=f)
                        cfunc = cr.entry_point
                        prec = 'single' if tx in (types.float32,\
                            types.complex64) else 'double'
                        for vx in x_values:
                            try:
                                expected = pyfunc(vx, xargs)
                            except ValueError as e:
                                self.assertIn("math domain error", str(e))
                                continue
                            got = cfunc(vx, xargs)
                            msg = 'for input %r with prec %r' % (vx, prec)
                            self.assertPreciseEqual(got, expected, prec=prec,
                                                    ulps=ulps, msg=msg,
                                                    ignore_sign_on_zero=
                                                    ignore_sign_on_zero,
                                                    abs_tol=abs_tol)

    def test_sinc(self, flags=no_pyobj_flags):
        """
        Tests the sinc() function.
        This test is purely to assert numerical computations are correct.
        """

        # Ignore sign of zeros, this will need masking depending on numpy
        # version once the fix to numpy complex division is in upstream
        # See: https://github.com/numpy/numpy/pull/6699
        isoz = True

        # Testing sinc(1.) leads to sin(pi)/pi, which is below machine
        # precision in practice on most machines. Small floating point
        # differences in sin() etc. may lead to large differences in the result
        # that are at a range that is inaccessible using standard width
        # floating point representations.
        # e.g. Assume float64 type.
        # sin(pi) ~= 1e-16, but should be zero
        # sin(pi)/pi ~= 1e-17, should be zero, error carried from above
        # float64 has log10(2^53)~=15.9 digits of precision and the magnitude
        # change in the alg is > 16  digits (1.0...0 -> 0.0...0),
        # so comparison via ULP is invalid.
        # We therefore opt to assume that values under machine precision are
        # equal in this case.
        tol="eps"

        pyfunc = sinc

        # real domain scalar context
        x_values = [1., -1., 0.0, -0.0, 0.5, -0.5, 5, -5, 5e-21, -5e-21]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        self.run_unary_real(pyfunc, x_types, x_values, flags,
                            ignore_sign_on_zero=isoz, abs_tol=tol)

        # real domain vector context
        x_values = np.array(x_values)
        x_types = [types.float32, types.float64]
        self.run_unary_real(pyfunc, x_types, x_values, flags=flags,
                               ignore_sign_on_zero=isoz, abs_tol=tol)

        # complex domain scalar context
        x_values = [1.+0j, -1+0j, 0.0+0.0j, -0.0+0.0j, 0+1j, 0-1j, 0.5+0.0j,
                    -0.5+0.0j, 0.5+0.5j, -0.5-0.5j, 5+5j, -5-5j,
                    # the following are to test sin(x)/x for small x
                    5e-21+0j, -5e-21+0j, 5e-21j, +(0-5e-21j)
                    ]
        x_types = [types.complex64, types.complex128] * (len(x_values) // 2)
        self.run_unary_complex(pyfunc, x_types, x_values, flags=flags,
                               ignore_sign_on_zero=isoz, abs_tol=tol, ulps=2)


        # complex domain vector context
        x_values = np.array(x_values)
        x_types = [types.complex64, types.complex128]
        self.run_unary_complex(pyfunc, x_types, x_values, flags=flags,
                               ignore_sign_on_zero=isoz, abs_tol=tol, ulps=2)


    def test_angle(self, flags=no_pyobj_flags):
        """
        Tests the angle() function.
        This test is purely to assert numerical computations are correct.
        """
        pyfunc = angle

        # real domain scalar context
        x_values = [1., -1., 0.0, -0.0, 0.5, -0.5, 5, -5]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        xtra_values = [True, False]
        xtra_types = [types.bool_, types.bool_]
        self.run_unary_real(pyfunc, x_types, x_values,
             flags, func_extra_types=xtra_types, func_extra_args=xtra_values)

        # real domain vector context
        x_values = np.array(x_values)
        x_types = [types.float32, types.float64]
        self.run_unary_real(pyfunc, x_types, x_values,
             flags, func_extra_types=xtra_types, func_extra_args=xtra_values)

        # complex domain scalar context
        x_values = [1.+0j, -1+0j, 0.0+0.0j, -0.0+0.0j, 1j, -1j, 0.5+0.0j,
                    -0.5+0.0j, 0.5+0.5j, -0.5-0.5j, 5+5j, -5-5j]
        x_types = [types.complex64, types.complex128] * (len(x_values) // 2)
        self.run_unary_complex(pyfunc, x_types, x_values,
              func_extra_types=xtra_types, func_extra_args=xtra_values,
              flags=flags)

        # complex domain vector context
        x_values = np.array(x_values)
        x_types = [types.complex64, types.complex128]
        self.run_unary_real(pyfunc, x_types, x_values,
             flags, func_extra_types=xtra_types, func_extra_args=xtra_values)
