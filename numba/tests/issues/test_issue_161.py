# -*- coding: utf-8 -*-

"""
>>> tuple_unpacking_error(2)
Traceback (most recent call last):
    ...
NumbaError: ...: Cannot unpack value of type int
"""

from __future__ import print_function, division, absolute_import

import numba
from numba import *
import numpy as np

@autojit(warnstyle="simple")
def tuple_unpacking_error(obj):
    a, b = obj
    print(a, b)

if __name__ == "__main__":
    numba.testing.testmod()
