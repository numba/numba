
from numba.np.npyimpl import registry as npyimpl_registry

def np_init():
    """Install the ``npyimpl`` registry on the default target context.
    Additionally registers NumPy-specific boxing/unboxing implementations.

    This is called by the CPU target during its initialization.
    """
    from numba.core.target_extension import (current_target,
                                             resolve_dispatcher_from_str)
    td = resolve_dispatcher_from_str(current_target()).targetdescr
    target_context = td.target_context
    target_context.install_registry(npyimpl_registry)

    # Register NumPy-specific boxing/unboxing implementations, which live
    # in the NumPy module.
    from numba.np import boxing # noqa F401
