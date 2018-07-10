"""
Expose top-level symbols that are safe for import *
"""
from __future__ import print_function, division, absolute_import

import platform
import re
import sys
import warnings

from . import config, errors, runtests, types

# Re-export typeof
from .special import typeof, prange, pndindex

# Re-export error classes
from .errors import *

# Re-export all type names
from .types import *

from .smartarray import SmartArray

# Re-export decorators
from .decorators import autojit, cfunc, generated_jit, jit, njit, stencil

# Re-export vectorize decorators
from .npyufunc import vectorize, guvectorize

# Re-export Numpy helpers
from .numpy_support import carray, farray, from_dtype

# Re-export jitclass
from .jitclass import jitclass

# Keep this for backward compatibility.
test = runtests.main


__all__ = """
    autojit
    cfunc
    from_dtype
    guvectorize
    jit
    jitclass
    njit
    stencil
    typeof
    prange
    stencil
    vectorize
    """.split() + types.__all__ + errors.__all__


_min_llvmlite_version = (0, 24, 0)
_min_llvm_version = (6, 0, 0)

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
    if pyver < (2, 7) or ((3,) <= pyver < (3, 4)):
        raise ImportError("Numba needs Python 2.7 or greater, or 3.4 or greater")

    np_version = numpy_support.version[:2]
    if np_version < (1, 7):
        raise ImportError("Numba needs Numpy 1.7 or greater")

def _try_enable_svml():
    """
    Tries to enable SVML if configuration permits use and the library is found.
    """
    if not config.DISABLE_INTEL_SVML:
        try:
            if sys.platform.startswith('linux'):
                llvmlite.binding.load_library_permanently("libsvml.so")
            elif sys.platform.startswith('darwin'):
                llvmlite.binding.load_library_permanently("libsvml.dylib")
            elif sys.platform.startswith('win'):
                llvmlite.binding.load_library_permanently("svml_dispmd")
            else:
                return False
            # The SVML library is loaded, therefore SVML *could* be supported.
            # Now see if LLVM has been compiled with the SVML support patch.
            # If llvmlite has the checking function `has_svml` and it returns
            # True, then LLVM was compiled with SVML support and the the setup
            # for SVML can proceed. We err on the side of caution and if the
            # checking function is missing, regardless of that being fine for
            # most 0.23.{0,1} llvmlite instances (i.e. conda or pip installed),
            # we assume that SVML was not compiled in. llvmlite 0.23.2 is a
            # bugfix release with the checking function present that will always
            # produce correct behaviour. For context see: #3006.
            try:
                if not getattr(llvmlite.binding.targets, "has_svml")():
                    # has detection function, but no svml compiled in, therefore
                    # disable SVML
                    return False
            except AttributeError:
                if platform.machine() == 'x86_64' and config.DEBUG:
                    msg = ("SVML was found but llvmlite >= 0.23.2 is "
                           "needed to support it.")
                    warnings.warn(msg)
                # does not have detection function, cannot detect reliably,
                # disable SVML.
                return False

            # All is well, detection function present and reports SVML is
            # compiled in, set the vector library to SVML.
            llvmlite.binding.set_option('SVML', '-vector-library=SVML')
            return True
        except:
            if platform.machine() == 'x86_64' and config.DEBUG:
                warnings.warn("SVML was not found/could not be loaded.")
    return False

_ensure_llvm()
_ensure_pynumpy()

# we know llvmlite is working as the above tests passed, import it now as SVML
# needs to mutate runtime options (sets the `-vector-library`).
import llvmlite

"""
Is set to True if Intel SVML is in use.
"""
config.USING_SVML = _try_enable_svml()

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
