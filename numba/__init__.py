"""
Expose top-level symbols that are safe for import *
"""
from __future__ import print_function, division, absolute_import
import re

from . import testing, decorators
from . import errors, special, types, config

# Re-export typeof
from .special import *
from .errors import *
from .pycc.decorators import export, exportmany

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

# Try to initialize cuda
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
""".split() + types.__all__ + special.__all__ + errors.__all__


_min_llvmlite_version = (0, 6, 0)

def _ensure_llvm():
    """
    Make sure llvmlite is operational.
    """
    import warnings
    import llvmlite

    # Only look at the the major, minor and bugfix version numbers.
    # Ignore other stuffs
    regex = re.compile(r'(\d+)\.(\d+).(\d+)')
    m = regex.match(llvmlite.__version__)
    if m:
        ver = tuple(map(int, m.groups()))
        if ver < _min_llvmlite_version:
            msg = ("Numba requires at least version %d.%d.%d of llvmlite.\n"
                   "Installed version is %s.\n"
                   "Please update llvmlite." %
                   (_min_llvmlite_version + (llvmlite.__version__,)))
            raise ImportError(msg)
    else:
        # Not matching?
        warnings.warn("llvmlite version format not recognized!")

    from llvmlite.binding import check_jit_execution
    check_jit_execution()


_ensure_llvm()


# Process initialization
# Should this be hooked into CPUContext instead?
from .targets.randomimpl import random_init
random_init()
del random_init

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
