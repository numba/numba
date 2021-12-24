import logging
import warnings

from numba.core.config import PYVERSION

if PYVERSION < (3, 9):
    try:
        from importlib_metadata import entry_points

    except ImportError as ex:
        raise ImportError(
            "backports.entry-points-selectable is required for Python version < 3.9, "
            "try:\n"
            "$ conda/pip install importlib.metadata"
        ) from ex
else:
    from importlib.metadata import entry_points

def _entry_points_sequence():
    raw_eps = entry_points()
    try:
        return raw_eps.select(group="numba_extensions", name="init")
    except AttributeError as e:
        return (item for item in eps.get("numba_extensions") if item.name=="init")


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

    for entry_point in _entry_points_sequence():
        logger.debug('Loading extension: %s', entry_point)
        try:
            func = entry_point.load()
            func()
        except Exception as e:
            msg = (f"Numba extension module '{entry_point.module}' "
                   f"failed to load due to '{type(e).__name__}({str(e)})'.")
            warnings.warn(msg, stacklevel=2)
            logger.debug('Extension loading failed for: %s', entry_point)
