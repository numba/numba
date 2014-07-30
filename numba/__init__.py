"""
Expose top-level symbols that are safe for import *
"""
from __future__ import print_function, division, absolute_import
import re
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


def _sentry_llvm_version():
    """
    Make sure we meet min llvmpy version
    """
    import sys
    import warnings
    import llvm
    min_version = (0, 12, 6)

    # Only look at the the major, minor and bugfix version numbers.
    # Ignore other stuffs
    regex = re.compile(r'(\d+)\.(\d+).(\d+)')
    m = regex.match(llvm.__version__)
    if m:
        ver = tuple(map(int, m.groups()))
        if ver < min_version:
            print("Numba requires at least version %d.%d.%d of llvmpy"
                  ".\nPlease update your version of llvmpy." % min_version)
            sys.exit()
    else:
        # Not matching?
        warnings.warns("llvmpy version format not recognized!")

_sentry_llvm_version()
