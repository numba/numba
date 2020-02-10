import logging

from numba.core.config import PYVERSION

if PYVERSION < (3, 8):
    try:
        import importlib_metadata
    except ImportError as ex:
        raise ImportError(
            "importlib_metadata backport is required for Python version < 3.8, "
            "try:\n"
            "$ conda/pip install importlib_metadata"
        ) from ex
else:
    from importlib import metadata as importlib_metadata


_already_initialized = False
logger = logging.getLogger(__name__)


def init_all():
    """Execute all `numba_extensions` entry points with the name `init`

    If extensions have already been initialized, this function does nothing.
    """
    global _already_initialized
    if _already_initialized:
        return

    # Must put this here to avoid extensions re-triggering initialization
    _already_initialized = True

    for entry_point in importlib_metadata.entry_points().get(
        "numba_extensions", tuple()
    ):
        if entry_point.name == "init":
            logger.debug("Loading extension: %s", entry_point)
            func = entry_point.load()
            func()
