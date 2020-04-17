import numba.core.errors as _errors
from numba.core.utils import PYVERSION as _PYVERSION

_moved_mod = "numba.core.errors"

if _PYVERSION >= (3, 7):
    __getattr__ = _errors.deprecate_moved_module_getattr(__name__, _moved_mod)
else:
    # wildcard import fails, import explicit
    from numba.core.errors import (  # noqa: F401
        WarningsFixer,
        TypingError,
        NumbaWarning,
        ForceLiteralArg,
    )

_errors.deprecate_moved_module(__name__, _moved_mod)
