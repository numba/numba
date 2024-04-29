import sys
from numba.core.utils import _RedirectSubpackage
from numba.core import config

if config.USE_LEGACY_TYPE_SYSTEM: # type: ignore
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.typed.old_listobject"
    )
else:
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.typed.new_listobject"
    )
