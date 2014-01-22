from __future__ import print_function, division, absolute_import
import sys
import os

# Debug flag to control compiler debug print
DEBUG = os.environ.get("NUMBA_DEBUG", False)

# Optimization level
OPT = os.environ.get("NUMBA_OPT", 2)

# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]
