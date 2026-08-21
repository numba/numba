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


# Install a unification method for Number types that
# uses NumPy's type promotion rules.
def unify_number(self, typingctx, other):
    """
    Unify the two number types using Numpy's rules.
    """
    from numba.np import numpy_support
    if isinstance(other, types.Number):
        # XXX: this can produce unsafe conversions,
        # e.g. would unify {int64, uint64} to float64
        a = numpy_support.as_dtype(self)
        b = numpy_support.as_dtype(other)
        sel = np.promote_types(a, b)
        return numpy_support.from_dtype(sel)


types.Number.unify = unify_number
