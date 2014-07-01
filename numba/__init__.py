"""
Expose top-level symbols that are safe for import *
"""
from __future__ import print_function, division, absolute_import
from . import testing, decorators
from ._version import get_versions
from . import special, types, config
# Re-export typeof
from .special import *
from .pycc.decorators import export, exportmany

# Version
__version__ = get_versions()['version']

# Re-export all type names
from .types import *

# Re export decorators
jit = decorators.jit
autojit = decorators.autojit
njit = decorators.njit

# Re export vectorize decorators
from .npyufunc import vectorize, guvectorize

# Re-export test entrypoint
test = testing.test

# Try initialize cuda
from . import cuda

__all__ = """
jit
autojit
njit
vectorize
guvectorize
export
exportmany
cuda
""".split() + types.__all__ + special.__all__
