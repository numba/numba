"""
Module to interact with Intel and other SPIR-V based devices
"""
from __future__ import print_function, absolute_import, division

from numba import config
import numba.testing
from dppy.ocldrv import *
from .device_init import *

def test(*args, **kwargs):
    if not is_available():
        dppy_error()

    return numba.testing.test("numba.dppy.tests", *args, **kwargs)
