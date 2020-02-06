import logging

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata

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

    for entry_point in importlib_metadata.entry_points().get(
        'numba_extensions', tuple()
    ):
        if entry_point.name == 'init':
            logger.debug('Loading extension: %s', entry_point)
            func = entry_point.load()
            func()
