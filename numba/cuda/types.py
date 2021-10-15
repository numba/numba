from numba.core import types


class Dim3(types.Type):
    """
    A 3-tuple (x, y, z) representing the position of a block or thread.
    """
    def __init__(self):
        super().__init__(name='Dim3')


class GridGroup(types.Type):
    """
    The grid of all threads in a cooperative kernel launch.
    """
    def __init__(self):
        super().__init__(name='GridGroup')


dim3 = Dim3()
grid_group = GridGroup()


class CUDADispatcher(types.Dispatcher):
    """The type of CUDA dispatchers"""
    # This type exists (instead of using types.Dispatcher as the type of CUDA
    # dispatchers) so that we can have an alternative lowering for them to the
    # lowering of CPU dispatchers - since the CPU target has first class
    # functions, its dispatchers lower to a constant address. However, we need
    # to lower to a dummy value since CUDA kernels and functions are not first
    # class functions.
