from __future__ import print_function, division, absolute_import
import sys
import os

# Debug flag to control compiler debug print
DEBUG = os.environ.get("NUMBA_DEBUG", False)

# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]
