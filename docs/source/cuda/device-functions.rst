
Writing Device Functions
========================

CUDA device functions can only be invoked from within the device (by a kernel
or device function).  To define a device function::

    from numba import cuda

    @cuda.jit(device=True)
    def a_devce_function(a, b):
        return a + b

Unlike a kernel function, a device function can return value like normal
functions.
