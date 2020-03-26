import numba.core.errors as _errors
from numba.core.utils import PYVERSION as _PYVERSION

# What was numba.unsafe.refcount got renamed to numba.core.unsafe
from numba.core import unsafe as refcount  # noqa: F401

_moved_mod = None
_errors.deprecate_moved_module(__name__, _moved_mod)
