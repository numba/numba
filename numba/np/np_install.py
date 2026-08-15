from numba.core.typing.context import Context
from numba.np import boxing   # noqa: F401

# Patch core registries to load additional npy registries
load_additional_core_registries = Context.load_additional_registries


def load_additional_npy_registries(self):
    from numba.np import arraydecl, npydecl  # noqa: F401, E501
    from numba.cffi import cffi_utils
    load_additional_core_registries(self)

    self.install_registry(npydecl.registry)
    self.install_registry(cffi_utils.registry)


Context.load_additional_registries = load_additional_npy_registries
