import sys
from numba.core.utils import _RedirectSubpackage
from numba.core import config

if config.USE_LEGACY_TYPE_SYSTEM: # type: ignore
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.core.types.old_misc"
    )
else:
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.core.types.new_misc"
    )
