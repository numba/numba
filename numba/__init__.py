"""
Expose top-level symbols that are safe for import *
"""
from __future__ import print_function, division, absolute_import

import re
import sys

from . import runtests, decorators
from . import errors, special, types, config

# Re-export typeof
from .special import *
from .errors import *

# Re-export all type names
from .types import *

from .smartarray import SmartArray

# Re export decorators
jit = decorators.jit
autojit = decorators.autojit
njit = decorators.njit
generated_jit = decorators.generated_jit

# Re export vectorize decorators
from .npyufunc import vectorize, guvectorize

# Re export from_dtype
from .numpy_support import from_dtype

# Re export jitclass
from .jitclass import jitclass

# Keep this for backward compatibility.
test = runtests.main


__all__ = """
jit
autojit
njit
vectorize
guvectorize
from_dtype
jitclass
""".split() + types.__all__ + special.__all__ + errors.__all__


_min_llvmlite_version = (0, 9, 0)
_min_llvm_version = (3, 7, 0)

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

    from llvmlite.binding import llvm_version_info, check_jit_execution

    if llvm_version_info < _min_llvm_version:
        msg = ("Numba requires at least version %d.%d.%d of LLVM.\n"
               "Installed llvmlite is built against version %d.%d.%d.\n"
               "Please update llvmlite." %
               (_min_llvm_version + llvm_version_info))
        raise ImportError(msg)

    check_jit_execution()


def _ensure_pynumpy():
    """
    Make sure Python and Numpy have supported versions.
    """
    import warnings
    from . import numpy_support

    pyver = sys.version_info[:2]
    strver = "%d.%d" % pyver
    if pyver in ((2, 6), (3, 3)):
        warnings.warn("Support for Python %d.%d will be dropped in Numba 0.25"
                      % pyver, DeprecationWarning)
    if pyver < (2, 6) or ((3,) <= pyver < (3, 3)):
        raise ImportError("Numba needs Python 2.6 or greater, or 3.3 or greater")

    np_version = numpy_support.version[:2]
    if np_version == (1, 6):
        warnings.warn("Support for Numpy %d.%d will be dropped in Numba 0.25"
                      % np_version, DeprecationWarning)
    if np_version < (1, 6):
        raise ImportError("Numba needs Numpy 1.6 or greater")


_ensure_llvm()
_ensure_pynumpy()


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
