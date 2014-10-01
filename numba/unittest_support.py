"""
This file fixes portability issues for unittest
"""
import sys
import warnings
from . import config

# Disable interpreter fallback upon import of this package
if config.COMPATIBILITY_MODE:
    warnings.warn("Unset INTERPRETER_FALLBACK")
    config.COMPATIBILITY_MODE = False

from numba.config import PYVERSION

if PYVERSION <= (2, 6):
    # Monkey-patch unittest2 into the import machinery, so that
    # submodule imports work properly too.
    import unittest2

    sys.modules['unittest'] = unittest2

from unittest import *
