
Writing Device Functions
========================

HSA device functions can only be invoked from a kernel
or another device function.  To define a device function::

    from numba import hsa

    @hsa.jit(device=True)
    def a_device_function(a, b):
        return a + b

Unlike a kernel function, a device function can return a value like normal
functions.
