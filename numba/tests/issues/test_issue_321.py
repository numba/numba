# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import *

f = autojit(lambda x: x * x)
assert f(10) == 100