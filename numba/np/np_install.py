from numba.core.typing.context import Context
from numba.np import boxing   # noqa: F401
from numba.core import types
import numpy as np

# Patch core registries to load additional npy registries
load_additional_core_registries = Context.load_additional_registries


def load_additional_npy_registries(self):
    from numba.np import arraydecl, npydecl  # noqa: F401, E501
    from numba.cffi import cffi_utils
    load_additional_core_registries(self)

    self.install_registry(npydecl.registry)
    self.install_registry(cffi_utils.registry)


Context.load_additional_registries = load_additional_npy_registries

# Patch the cast_python_value methods of Integer, Float, and Complex
# to use NumPy's casting functions instead of Python's built-in types.
npy_scalar_cast = lambda self, value: getattr(np, self.name)(value)

types.Integer.cast_python_value = npy_scalar_cast
types.Float.cast_python_value = npy_scalar_cast
types.Complex.cast_python_value = npy_scalar_cast
