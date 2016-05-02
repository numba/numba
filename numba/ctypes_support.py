"""
This file fixes portability issues for ctypes.
"""

from __future__ import absolute_import
from numba.config import PYVERSION
from ctypes import *

if PYVERSION <= (2, 7):
    c_ssize_t = {
        4: c_int32,
        8: c_int64,
    }[sizeof(c_size_t)]

