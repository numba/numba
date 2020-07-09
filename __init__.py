"""
Module to interact with Intel and other SPIR-V based devices
"""
from __future__ import print_function, absolute_import, division

from numba import config
import numba.testing

from numba.dppy_config import *
if dppy_present:
    from .device_init import *
else:
    raise ImportError("Importing dppy failed")

def test(*args, **kwargs):
    if dppy_present and not is_available():
        dppy_error()

    return numba.testing.test("numba.dppy.tests", *args, **kwargs)
