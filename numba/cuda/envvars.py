import os
import warnings

from numba.core import errors


def get_numbapro_envvar(envvar, default=None):
    # use vanilla get here so as to use `None` as a signal for not-set
    value = os.environ.get(envvar)
    if value is not None:
        url = ("http://numba.pydata.org/numba-doc/latest/cuda/overview.html",
               "#cudatoolkit-lookup")
        msg = ("\nEnvironment variables with the 'NUMBAPRO' prefix are "
               "deprecated and consequently ignored, found use of %s=%s.\n\n"
               "For more information about alternatives visit: %s"
               % (envvar, value, url))
        warnings.warn(errors.NumbaWarning(msg))
    return default


def get_numba_envvar(envvar, default=None):
    """Tries to load an environment variable with numba ``PREFIX + envvar``.
    Only the "NUMBA_" prefix is attempted for use herein. The use of the
    "NUMBAPRO_" prefix was deprecated in 0.45 with support removed in 0.46.
    However it is still checked solely to warn users that it has no effect.
    """
    assert not envvar.startswith('NUMBA')
    value = os.environ.get('NUMBA_' + envvar)
    if value is None:
        return get_numbapro_envvar('NUMBAPRO_' + envvar, default=default)
    else:
        return default
