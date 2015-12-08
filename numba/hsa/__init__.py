"""
Module that deals with HSA in a high level way
"""
from __future__ import print_function, absolute_import, division
import numba.testing
from .api import *
from .stubs import atomic
from . import initialize


def is_available():
    """Returns a boolean to indicate the availability of a HSA runtime.

    This will force initialization of the driver if it hasn't been
    initialized.
    """
    from .hsadrv.driver import hsa
    return hsa.is_available


def test(*args, **kwargs):
    if not is_available():
        raise RuntimeError("HSA is not detected")

    return numba.testing.test("numba.hsa.tests", *args, **kwargs)
