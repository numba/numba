
Writing Device Functions
========================

HSA device functions are functions that will run on the device but can only be
invoked from a kernel or another device function. Unlike a kernel function, a
device function can return a value like normal functions. To define a device
function the kwarg ``device`` must be set to ``True`` in the ``roc.jit``
decorator::

    from numba import roc

    @roc.jit(device=True)
    def a_device_function(a, b):
        return a + b

An example of using a device function::

    from numba import roc
    import numpy as np

    @roc.jit
    def kernel(an_array):
        pos = roc.get_global_id(0)
        if pos < an_array.size:  # Check array boundaries
            an_array[pos] = a_device_function(1, pos) # call device function

    @roc.jit(device = True)
    def a_device_function(a, b):
        return a + b

    n = 16
    x = np.zeros(n)

    kernel[1, n](x)

    print(x)

