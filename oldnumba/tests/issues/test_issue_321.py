# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys
from numba import *

meta = None
if sys.version_info[:2] > (2, 6):
    try:
        import meta
    except ImportError:
        pass

if meta:
    f = autojit(lambda x: x * x)
    assert f(10) == 100
