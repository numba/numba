# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys
from numba import *

if sys.version_info[:2] > (2, 6):
    f = autojit(lambda x: x * x)
    assert f(10) == 100