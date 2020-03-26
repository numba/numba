import numba.core.errors as _errors
from numba.core.utils import PYVERSION as _PYVERSION

_moved_mod = "numba.cpython.unicode"

if _PYVERSION >= (3, 7):
    __getattr__ = _errors.deprecate_moved_module_getattr(__name__, _moved_mod)
else:
    from numba.cpython.unicode import *  # noqa: F403, F401
    from numba.cpython.unicode import (  # noqa: F401
        _slice_span,
        _normalize_slice,
        _empty_string,
    )
    from numba.cpython.unicode import (  # noqa: F401
        PY_UNICODE_1BYTE_KIND,
        PY_UNICODE_2BYTE_KIND,
        PY_UNICODE_4BYTE_KIND,
        PY_UNICODE_WCHAR_KIND,
    )

_errors.deprecate_moved_module(__name__, _moved_mod)
