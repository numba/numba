from numba import njit, types
from numba.extending import overload


import mymodule


@overload(mymodule.set_to_x)
def set_to_x_jit(arr, x):

    if not isinstance(arr, types.Array):
        return
    if not isinstance(x, types.Integer):
        return

    def set_to_x_impl(arr, x):
        arr[:] = x

    return set_to_x_impl


@njit
def myalgorithm(a, x):
    # algorithm code
    mymodule.set_to_x(a, x)
    # algorithm code
