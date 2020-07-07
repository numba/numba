import sys
from numba.core.errors import _MovedModule

from numba.misc import quicksort

sys.modules[__name__] = _MovedModule(locals(), None)
sys.modules[__name__].quicksort = quicksort

