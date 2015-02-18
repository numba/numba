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

# Re export from_dtype
from .numpy_support import from_dtype

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
from_dtype
""".split() + types.__all__ + special.__all__


def _sentry_llvm_version():
    """
    Make sure we meet min llvmlite version
    """
    import warnings
    import llvmlite
    min_version = (0, 1, 0)

    # Only look at the the major, minor and bugfix version numbers.
    # Ignore other stuffs
    regex = re.compile(r'(\d+)\.(\d+).(\d+)')
    m = regex.match(llvmlite.__version__)
    if m:
        ver = tuple(map(int, m.groups()))
        if ver < min_version:
            msg = ("Numba requires at least version %d.%d.%d of llvmlite.\n"
                   "Installed version is %s.\n"
                   "Please update llvmlite." %
                   (min_version + (llvmlite.__version__,)))
            raise ImportError(msg)
    else:
        # Not matching?
        warnings.warn("llvmlite version format not recognized!")

_sentry_llvm_version()


# Process initialization
# Should this be hooked into CPUContext instead?
from .targets.randomimpl import random_init
random_init()
del random_init
