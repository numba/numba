import signal
import sys
from numba import njit
import numpy as np

def sigterm_handler(signum, frame):
    raise RuntimeError("Caught SIGTERM")

def busy_func(a, b, q=None):
    sys.stdout.flush()
    sys.stderr.flush()
    signal.signal(signal.SIGTERM, sigterm_handler)
    try:
        busy_func_inner(a, b)
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException as e:
        if q is not None:
            q.put(e)

@njit(parallel=True)
def busy_func_inner(a, b):
    c = a + b * np.sqrt(a) + np.sqrt(b)
    d = np.sqrt(a + b * np.sqrt(a) + np.sqrt(b))
    return c + d
