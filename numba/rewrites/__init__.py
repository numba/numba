import numba.core.errors as _errors
from numba.core.utils import PYVERSION as _PYVERSION

_moved_mod = "numba.core.rewrites"

if _PYVERSION >= (3, 7):
    __getattr__ = _errors.deprecate_moved_module_getattr(__name__, _moved_mod)
else:
    from numba.core.rewrites.registry import (
        register_rewrite,  # noqa: F401
        rewrite_registry,  # noqa: F401
        Rewrite,
    )  # noqa: F401

_errors.deprecate_moved_module(__name__, _moved_mod)
