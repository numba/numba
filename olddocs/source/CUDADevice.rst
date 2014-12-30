CUDA Device Management
=======================

For multi-GPU machines, users may want to select which GPU to use.
By default the CUDA driver selects the fastest GPU as the device 0,
which is the default device used by Numba.

The features introduced on this page are generally not of interest
unless working with systems hosting/offering more than one
CUDA-capable GPU.

Device Selection
----------------

Device selection must be done before any CUDA feature is used.

::

    from numba import cuda
    cuda.select_device(0)

.. autofunction:: numba.cuda.select_device


The device can be closed by:

::

    cuda.close()

.. autofunction:: numba.cuda.close


Users can then create a new context with another device.

::

    cuda.select_device(1)  # assuming we have 2 GPUs


.. NOTE:: Compiled functions are associated with the CUDA context.
    This makes it not very useful to close and create new devices,
    though it is certainly useful for choosing which device to use when the machine
    has multiple GPUs.


.. Future feature that needs more polishing.


    Multi-Device
    -------------

    It is possible to use multiple devices by using multiple threads and
    associating different devices to different threads.

    .. NOTE::  The compute mode of a device can be configured to be
    exclusive to a thread or process.  This prevents the user from creating
    multiple context on the same device in different threads.  The solution is to
    use the `nvidia-smi` commandline tool to query and modify the compute mode.
    Refer to the documentation in `nvidia-smi`.
