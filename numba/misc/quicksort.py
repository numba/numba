import sys
from numba.core.utils import _RedirectSubpackage
from numba.core import config

if config.USE_LEGACY_TYPE_SYSTEM:
    sys.modules[__name__] = _RedirectSubpackage(locals(), "numba.misc.old_quicksort")
else:
    sys.modules[__name__] = _RedirectSubpackage(locals(), "numba.misc.new_quicksort")
