"""
NumPy extensions.
"""
from numba.np.numpy_support import numpy_version

if numpy_version < (2, 5):
    from numba.np.arraymath import cross2d

    __all__ = [
        'cross2d'
    ]
