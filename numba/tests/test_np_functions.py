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

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

no_pyobj_flags = Flags()

def sinc(x):
    return np.sinc(x)

class TestNPFunctions(TestCase):
    """
    Contains tests and test helpers for numpy methods the are of type
    "class< 'function' >.
    """

    def setUp(self):
        self.ccache = CompilationCache()

    def run_unary_real(self, pyfunc, x_types, x_values,
        flags=enable_pyobj_flags, prec='exact', **kwargs):
        """
        Runs tests for a unary function operating in the numerical real space.

        Parameters
        ----------
        pyfunc : a python function definition holding that calls the numpy
                 functions to be tested.
        x_types: the types of the values being tested, see numba.types
        x_values: the numerical values of the values to be tested
        flags: flags to pass to the ComplicationnCache::ccache::compile function
        prec: the required precision match, see assertPreciseEqual

        Notes:
        ------
        x_types and x_values must have the same length

        """
        for tx, vx in zip(x_types, x_values):
            cr = self.ccache.compile(pyfunc, (tx,), flags=flags)
            cfunc = cr.entry_point
            got = cfunc(vx)
            expected = pyfunc(vx)
            actual_prec = 'single' if tx is types.float32 else prec
            msg = 'for input %r with prec %r' % (vx, prec)
            self.assertPreciseEqual(got, expected, prec=actual_prec, msg=msg,
                                    **kwargs)

    def run_unary_complex(self, pyfunc, x_types, x_values, ulps=1,
                  flags=enable_pyobj_flags):
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
        flags: flags to pass to the ComplicationnCache::ccache::compile function
        prec: the required precision match, see assertPreciseEqual

        Notes:
        ------
        x_types and x_values must have the same length
        """
        for tx in x_types:
            cr = self.ccache.compile(pyfunc, (tx,), flags=flags)
            cfunc = cr.entry_point
            prec = 'single' if tx in (types.float32, types.complex64) else \
            'double'
            for vx in x_values:
                try:
                    expected = pyfunc(vx)
                except ValueError as e:
                    self.assertIn("math domain error", str(e))
                    continue
                got = cfunc(vx)
                msg = 'for input %r with prec %r' % (vx, prec)
                self.assertPreciseEqual(got, expected, prec=prec,
                                        ulps=ulps, msg=msg)

    def test_sinc(self, flags=enable_pyobj_flags):
        """
        Tests the sinc() function. The array context test for the sinc()
        function is performed in the standard ufuncs test.
        This test is purely to assert numerical computations are correct.
        """
        pyfunc = sinc
        x_values = [1., -1., 0.0, -0.0, 0.5, -0.5, 5, -5, 5e-21, -5e-21]
        x_types = [types.float32, types.float64] * (len(x_values) // 2)
        self.run_unary_real(pyfunc, x_types, x_values, flags)

        x_values = [1.+0j, -1+0j, 0.0+0.0j, -0.0+0.0j, 1j, -1j, 0.5+0.0j,
                    -0.5+0.0j, 0.5+0.5j, -0.5-0.5j, 5+5j, -5-5j,
                    # the following are to test sin(x)/x for small x
                    5e-21+0j, -5e-21+0j, 5e-21j, -5e-21j]
        x_types = [types.complex64, types.complex128] * (len(x_values) // 2)
        self.run_unary_complex(pyfunc, x_types, x_values, flags=flags)
