import logging
import warnings

from pkg_resources import iter_entry_points

_already_initialized = False
logger = logging.getLogger(__name__)


def init_all():
    '''Execute all `numba_extensions` entry points with the name `init`

    If extensions have already been initialized, this function does nothing.
    '''
    global _already_initialized
    if _already_initialized:
        return

    # Must put this here to avoid extensions re-triggering initialization
    _already_initialized = True

    for entry_point in iter_entry_points('numba_extensions', 'init'):
        logger.debug('Loading extension: %s', entry_point)
        try:
            func = entry_point.load()
            func()
        except Exception as e:
            msg = "Numba extension module '{}' failed to load due to '{}({})'."
            warnings.warn(msg.format(entry_point.module_name, type(e).__name__,
                                     str(e)), stacklevel=2)
            logger.debug('Extension loading failed for: %s', entry_point)
