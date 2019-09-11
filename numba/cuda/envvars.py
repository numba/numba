import os


def get_numba_envvar(envvar, default=None):
    """Tries to load an environment variable matchin ``PREFIX + envvar``.
    Only the "NUMBA_" prefix is attempted herein.
    """
    assert not envvar.startswith('NUMBA')
    return os.environ.get('NUMBA_' + envvar, default)
