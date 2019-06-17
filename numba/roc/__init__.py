"""
Module that deals with HSA in a high level way
"""
from __future__ import print_function, absolute_import, division
import os
import numba.testing
from .api import *
from .stubs import atomic

def is_available():
    """Returns a boolean to indicate the availability of a HSA runtime.

    This will force initialization of the driver if it hasn't been
    initialized. It also checks that a toolchain is present.
    """
    from .hsadrv.driver import hsa

    from .hlc import hlc, libhlc
    has_a_toolchain = False

    try:
       libhlc.HLC()
       has_a_toolchain = True
    except:
        try:
            cmd = hlc.CmdLine().check_tooling()
            has_a_toolchain = True
        except:
            pass

    return hsa.is_available and has_a_toolchain

if is_available():
    from .hsadrv.driver import hsa
    agents = list(hsa.agents)
else:
    agents = []

def test(*args, **kwargs):
    if not is_available():
        raise RuntimeError("HSA is not detected")

    return numba.testing.test("numba.hsa.tests", *args, **kwargs)
