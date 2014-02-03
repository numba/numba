"""
This file fixes portability issues for unittest
"""

from numba.config import PYVERSION

if PYVERSION <= (2, 6):
    from unittest2 import *
else:
    from unittest import *
