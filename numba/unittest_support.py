"""
This file fixes portability issues for unittest
"""
import sys
import warnings
from . import config

from numba.config import PYVERSION

from unittest import *
from unittest import case