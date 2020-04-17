import numba.core.errors as _errors
from numba.core.utils import PYVERSION as _PYVERSION

_moved_mod = "numba.cpython.hashing"

if _PYVERSION >= (3, 7):
    __getattr__ = _errors.deprecate_moved_module_getattr(__name__, _moved_mod)
else:
    from numba.cpython.hashing import *  # noqa: F403, F401
    from numba.cpython.hashing import _Py_hash_t  # noqa: F403, F401

_errors.deprecate_moved_module(__name__, _moved_mod)
