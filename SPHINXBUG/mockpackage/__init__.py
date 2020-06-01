import sys
sys.path.insert(0, ".")
print(sys.path)
import numba
from functools import lru_cache
import inspect

inspect.isfunction()
def f_python(x):
    """
    I am a python function
    """
    return x

@numba.njit()
def f_njit(x):
    """
    I am an njit function
    """
    return x

@numba.vectorize([numba.float64(numba.float64)], nopython=True)
def f_vectorize(x):
    """
    I am a vectorize function
    """
    return x

@lru_cache()
def f_cache(x):
    "I am cached"
    return x