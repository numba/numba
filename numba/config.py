from __future__ import print_function, division, absolute_import
import sys
import os

# Debug flag to control compiler debug print
DEBUG = int(os.environ.get("NUMBA_DEBUG", '0'))

# JIT Debug flag to trigger IR instruction print
DEBUG_JIT = int(os.environ.get("NUMBA_DEBUG_JIT", '0'))

# Optimization level
OPT = int(os.environ.get("NUMBA_OPT", '3'))

# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]
