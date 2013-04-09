import numba
from numba import *

@autojit
def test_object_iteration(obj):
    """
    >>> test_object_iteration([1, 2, 3])
    1
    2
    3
    """
    for x in obj:
        print(x)

if __name__ == '__main__':
#    test_object_iteration([1, 2, 3])
    numba.testing.testmod()
