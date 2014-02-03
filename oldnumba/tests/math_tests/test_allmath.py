# -*- coding: utf-8 -*-

"""
Test all support math functions
"""

from __future__ import print_function, division, absolute_import

import math
import cmath
import collections
from itertools import chain

import numba as nb
import numpy as np
from numpy.testing import assert_almost_equal

# ______________________________________________________________________
# Common

def run_common(mod, x):
    "np, math and cmath"
    y0  = mod.sin(x)
    y1  = mod.cos(x)
    y2  = mod.tan(x)
    y3  = mod.sqrt(x)
    y4  = mod.sinh(x)
    y5  = mod.cosh(x)
    y6  = mod.tanh(x)
    y7  = mod.log(x)
    y8  = mod.log10(x)
    y9  = mod.exp(x)
    return (y0, y1, y2, y3, y4, y5, y6, y7, y8, y9)

def run_np_math(mod, x):
    "np (floating, complex) and math (floating)"
    if hasattr(mod, 'expm1'):
        y0  = mod.expm1(x)
    else:
        y0 = 0.0
    y1  = mod.log1p(x)
    return (y0, y1)

def run_commonf(mod, x):
    "np and math"
    y0  = mod.floor(x)
    y1  = mod.ceil(x)
    y2  = mod.hypot(x, x)
    return (y0, y1, y2)

# ______________________________________________________________________
# NumPy

def run_np_arc(mod, x):
    "np only"
    y0  = mod.arccos(x)
    y1  = mod.arcsin(x)
    y2  = mod.arctan(x)
    y3  = mod.arcsinh(x)
    y4  = mod.arccosh(1.0/x)
    return (y0, y1, y2, y3, y4)

def run_np_misc(mod, x):
    "np only"
    y0  = mod.log2(x)
    y1  = mod.exp2(x)
    y2  = mod.rint(x)
    y3  = mod.power(x, x)
    y4  = mod.absolute(x * -1) # TODO: USub for some types
    return (y0, y1, y2, y3, y4)

# ______________________________________________________________________
# Python

def run_py_arc(mod, x):
    "math and cmath"
    y0  = mod.acos(x)
    y1  = mod.asin(x)
    y2  = mod.atan(x)
    y3  = mod.asinh(x)
    y4  = mod.acosh(1.0/x)
    y5  = mod.atanh(x)
    return (y0, y1, y2, y3, y4, y5)

def misc_floating(mod, x):
    "miscellaneous"
    # y0  = math.erfc(x)

    y1  = math.atan2(x, x)
    y2  = np.arctan2(x, x)
    y3  = np.logaddexp(x, x)
    y4  = np.logaddexp2(x, x)
    return (y1, y2, y3, y4) #(y0, y1)

# ______________________________________________________________________
# Run tests

Suite = collections.namedtuple('Suite', ['mod', 'types'])
merge = lambda d1, d2: dict(chain(d1.items(), d2.items()))

integral = nb.short, nb.int_, nb.uint, nb.long_, nb.ulong, nb.longlong, nb.ulonglong
floating = nb.float_, nb.double, #nb.longdouble
complexes = nb.complex64, nb.complex128, #nb.complex256

fdata     = { integral : 6, floating: 6.0, (nb.object_,): 6.0 }
cdata     = { complexes: 6.0+4.0j }
data      = merge(fdata, cdata)

arc_fdata = { floating: 0.6, (nb.object_,): 0.6 }
arc_cdata = { complexes: 0.6+0.4j }
arc_data  = merge(arc_fdata, arc_cdata)

tests = {
    run_common  : [Suite(math, fdata), Suite(cmath, cdata)],
    run_np_math : [Suite(np, data), Suite(math, fdata)],
    run_commonf : [Suite(np, fdata), Suite(math, fdata)],
    run_np_arc  : [Suite(np, arc_data)],
    run_py_arc  : [Suite(math, arc_fdata), Suite(cmath, arc_cdata)],
    run_np_misc : [Suite(np, data)],
    # run_py_math : [Suite(math, fdata)],
    misc_floating : [Suite(math, fdata)]
}

def run():
    for test, suites in tests.iteritems():
        for suite in suites:
            for types, data in suite.types.iteritems():
                for ty in types:
                    print("running:", test.__name__, ty)
                    signature = nb.object_(nb.typeof(suite.mod), ty)
                    jitter = nb.jit(signature)
                    jitted = jitter(test)

                    r1 = test(suite.mod, data)
                    r2 = jitted(suite.mod, data)
                    # print(r1, r2)
                    assert np.allclose(r1, r2)

                    r3 = jitted(suite.mod, data)
                    assert np.allclose(r1, r3)
run()
