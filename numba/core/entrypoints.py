import logging
import re
import warnings

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
        "numba_extensions", ()
    ):
        if entry_point.name == "init":
            logger.debug('Loading extension: %s', entry_point)
            try:
                func = entry_point.load()
                func()
            except Exception as e:
                # entry_point.module was only added in Python 3.9. The
                # backported importlib_metadata (used on Python 3.7) also adds
                # it. So, on Python 3.8 we need to duplicate the logic used to
                # extract the module name.
                if PYVERSION == (3, 8):
                    pattern = re.compile(
                        r'(?P<module>[\w.]+)\s*'
                        r'(:\s*(?P<attr>[\w.]+))?\s*'
                        r'(?P<extras>\[.*\])?\s*$'
                    )
                    match = pattern.match(entry_point.value)
                    module = match.group('module')
                else:
                    module = entry_point.module
                msg = (f"Numba extension module '{module}' "
                       f"failed to load due to '{type(e).__name__}({str(e)})'.")
                warnings.warn(msg, stacklevel=2)
                logger.debug('Extension loading failed for: %s', entry_point)
