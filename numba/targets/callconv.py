import sys
from numba.core.errors import _MovedModule
sys.modules[__name__] = _MovedModule(locals(), "numba.core.callconv")
