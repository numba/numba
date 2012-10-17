import os
from ctypes import *
import subprocess

from numba.pycc import find_shared_ending
from numba.pycc import pycc

modulename = os.path.join(os.path.dirname(__file__), 'compile_with_pycc')
pycc.main(args=[modulename + '.py'])
lib = CDLL(modulename + find_shared_ending())

lib.mult.argtypes = [c_double, c_double]
lib.mult.restype = c_double

lib.multf.argtypes = [c_float, c_float]
lib.multf.restype = c_float


res = lib.mult(123, 321)
print 'lib.mult(123, 321) =', res
assert res == 123 * 321


res = lib.multf(987, 321)
print 'lib.multf(987, 321) =', res
assert res == 987 * 321


