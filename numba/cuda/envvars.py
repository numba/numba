import os
import warnings

from numba import errors


def get_numbapro_envvar(envvar, default=None):
    # use vanilla get here so as to use `None` as a signal for not-set
    value = os.environ.get(envvar)
    if value is not None:
        url = ("http://numba.pydata.org/numba-doc/latest/reference/"
               "deprecation.html#deprecation-of-numbapro-environment-variables")
        msg = ("\nEnvironment variables with the 'NUMBAPRO' prefix are "
               "deprecated, found use of %s=%s.\n\nFor more information visit "
               "%s" % (envvar, value, url))
        warnings.warn(errors.NumbaDeprecationWarning(msg))
        return value
    else:
        return default


def get_numba_envvar(envvar, default=None):
    """Tries to load an environment variable with numba ``PREFIX + envvar``.
    Two prefixes are tried.  First "NUMBA_". Then, "NUMBAPRO_".
    """
    assert not envvar.startswith('NUMBA')
    value = os.environ.get('NUMBA_' + envvar)
    if value is None:
        return get_numbapro_envvar('NUMBAPRO_' + envvar, default=default)
    else:
        return default
