import sys
sys.path.insert(0, ".")
print(sys.path)
import numba

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
