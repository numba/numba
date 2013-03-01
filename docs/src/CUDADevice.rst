CUDA Device Management
=======================

For multi-GPU machines, users may want to select which GPU to use.
By default the CUDA driver selects the fastest GPU as the device 0,
which is the default device used by NumbaPro.

The features introduced on this page is optional for most users.

Device Selection
----------------

Device selection must be done before any cuda feature is used.

::

    from numbapro import cuda
    cuda.select_device(0)

This creates a new CUDA context with the selected device.
The context is associated with the current thread.
NumbaPro currently allows only one context per thread.

The device can be closed by:

::

    cuda.close()

This releases the CUDA context from the current thread.

Users can than create a new context with another device.

::

    cuda.select_device(1)  # assuming we have 2 GPUs


**Note:** Compiled functions are associated with the context CUDA context.
This makes it not very useful to close and create new devices.
But, it is certainly useful for choosing which device to use when the machine
has multiple GPUs.


.. Future feature that needs more polishing.


    Multi-Device
    -------------

    It is possible to use multiple devices by using multiple threads and
    associating different devices to different threads.

    **NOTE:**  The compute mode of a device can be configured to be
    exclusive to a thread or process.  This prevents the user from creating
    multiple context on the same device in different threads.  The solution is to
    use the `nvidia-smi` commandline tool to query and modify the compute mode.
    Refer to the documentation in `nvidia-smi`.
