# -*- coding: utf-8 -*-

# The pycc module requires setuptools.
try:
    import setuptools
except ImportError:
    msg = "The 'setuptools' package is required at runtime for pycc support."
    raise ImportError(msg)

# Public API
from .cc import CC
from .decorators import export, exportmany
