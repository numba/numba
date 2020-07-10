"""
Module to interact with Intel and other SPIR-V based devices
"""
from __future__ import print_function, absolute_import, division

from numba import config
import numba.testing

from numba.dppl_config import *
if dppl_present:
    from .device_init import *
else:
    raise ImportError("Importing dppl failed")

def test(*args, **kwargs):
    if not dppl_present and not is_available():
        dppl_error()

    return numba.testing.test("numba.dppl.tests", *args, **kwargs)
