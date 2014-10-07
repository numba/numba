"""
This file fixes portability issues for unittest
"""
import sys
import warnings
from . import config

from numba.config import PYVERSION

if PYVERSION <= (2, 6):
    # Monkey-patch unittest2 into the import machinery, so that
    # submodule imports work properly too.
    import unittest2

    sys.modules['unittest'] = unittest2

from unittest import *
