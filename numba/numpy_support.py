import numba.core.errors as _errors
from types import ModuleType

_moved_mod = "numba.np.numpy_support"

class module(ModuleType):
    __getattr__ = _errors.deprecate_moved_module_getattr(__name__, _moved_mod)

sys.modules[__name__] = module(__name__)
