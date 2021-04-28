import importlib


def __getattr__(name):
    """Redirect to numba.core.types
    """
    # Uses PEP-562 but requires python>3.6
    basemod = importlib.import_module(f"numba.core.types", __name__)
    return getattr(basemod, name)
