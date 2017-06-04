from __future__ import print_function, absolute_import, division

from numba import config
import numba.testing

from .device_init import *
from .device_init import _auto_device


def test(*args, **kwargs):
    if not is_available():
        raise ocl_error()

    return numba.testing.test("numba.ocl.tests", *args, **kwargs)
