from numba.core.typing.context import Context

# Patch core registries to load additional npy registries
load_additional_core_registries = Context.load_additional_registries


def load_additional_npy_registries(self):
    from numba.np import arraydecl, npydecl  # noqa: F401, E501
    self.install_registry(npydecl.registry)


Context.load_additional_registries = load_additional_npy_registries
