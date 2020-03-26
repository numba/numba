import numba.core.errors as _errors
from numba.core.utils import PYVERSION as _PYVERSION

_moved_mod = "numba.core.inline_closurecall"

if _PYVERSION >= (3, 7):
    __getattr__ = _errors.deprecate_moved_module_getattr(__name__, _moved_mod)
else:
    from numba.core.inline_closurecall import *  # noqa: F403, F401
    from numba.core.inline_closurecall import _replace_returns  # noqa: F401
    from numba.core.inline_closurecall import _replace_freevars  # noqa: F401
    from numba.core.inline_closurecall import _add_definitions  # noqa: F401

_errors.deprecate_moved_module(__name__, _moved_mod)
