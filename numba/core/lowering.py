import sys
from numba.core.utils import _RedirectSubpackage
from numba.core import config

if config.USE_LEGACY_TYPE_SYSTEM: # type: ignore
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.core.old_lowering"
    )
else:
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.core.new_lowering"
    )
