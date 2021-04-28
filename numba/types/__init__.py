import importlib


def __getattr__(name):
    # Uses PEP-562 but requires python>3.6
    try:
        return importlib.import_module(f"numba.core.types.{name}", __name__)
    except ModuleNotFoundError:
        raise AttributeError