"""
Expose top-level symbols that are safe for import *
"""

import platform
import re
import sys
import warnings

from ._version import get_versions
from numba.misc.init_utils import generate_version_info

__version__ = get_versions()['version']
version_info = generate_version_info(__version__)
del get_versions
del generate_version_info


from numba.core import config
from numba.testing import _runtests as runtests
from numba.core import types, errors

# Re-export typeof
from numba.misc.special import (
    typeof, prange, pndindex, gdb, gdb_breakpoint, gdb_init,
    literally, literal_unroll,
)

# Re-export error classes
from numba.core.errors import *

# Re-export types itself
import numba.core.types as types

# Re-export all type names
from numba.core.types import *

# Re-export decorators
from numba.core.decorators import (cfunc, generated_jit, jit, njit, stencil,
                                   jit_module)

# Re-export vectorize decorators and the thread layer querying function
from numba.np.ufunc import (vectorize, guvectorize, threading_layer,
                            get_num_threads, set_num_threads)

# Re-export Numpy helpers
from numba.np.numpy_support import carray, farray, from_dtype

# Re-export experimental
from numba import experimental

# Initialize withcontexts
import numba.core.withcontexts
from numba.core.withcontexts import objmode_context as objmode

# Keep this for backward compatibility.
test = runtests.main


__all__ = """
    cfunc
    from_dtype
    guvectorize
    jit
    experimental
    njit
    stencil
    jit_module
    jitclass
    typeof
    prange
    gdb
    gdb_breakpoint
    gdb_init
    vectorize
    objmode
    literal_unroll
    get_num_threads
    set_num_threads
    """.split() + types.__all__ + errors.__all__


_min_llvmlite_version = (0, 33, 0)
_min_llvm_version = (9, 0, 0)

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

def _ensure_critical_deps():
    """
    Make sure Python, NumPy and SciPy have supported versions.
    """
    from numba.np.numpy_support import numpy_version
    from numba.core.utils import PYVERSION

    if PYVERSION < (3, 6):
        raise ImportError("Numba needs Python 3.6 or greater")

    if numpy_version < (1, 15):
        raise ImportError("Numba needs NumPy 1.15 or greater")

    try:
        import scipy
    except ImportError:
        pass
    else:
        sp_version = tuple(map(int, scipy.__version__.split('.')[:2]))
        if sp_version < (1, 0):
            raise ImportError("Numba requires SciPy version 1.0 or greater")


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
_ensure_critical_deps()

# we know llvmlite is working as the above tests passed, import it now as SVML
# needs to mutate runtime options (sets the `-vector-library`).
import llvmlite

"""
Is set to True if Intel SVML is in use.
"""
config.USING_SVML = _try_enable_svml()


# ---------------------- WARNING WARNING WARNING ----------------------------
# The following imports occur below here (SVML init) because somewhere in their
# import sequence they have a `@njit` wrapped function. This triggers too early
# a bind to the underlying LLVM libraries which then irretrievably sets the LLVM
# SVML state to "no SVML". See https://github.com/numba/numba/issues/4689 for
# context.
# ---------------------- WARNING WARNING WARNING ----------------------------

# Initialize typed containers
import numba.typed
