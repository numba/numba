
Device management
=================

For multi-GPU machines, users may want to select which GPU to use.
By default the CUDA driver selects the fastest GPU as the device 0,
which is the default device used by Numba.

The features introduced on this page are generally not of interest
unless working with systems hosting/offering more than one CUDA-capable GPU.

Device Selection
----------------

If at all required, device selection must be done before any CUDA feature is
used.

::

    from numba import cuda
    cuda.select_device(0)

The device can be closed by:

::

    cuda.close()

Users can then create a new context with another device.

::

    cuda.select_device(1)  # assuming we have 2 GPUs


.. autofunction:: numba.cuda.select_device


.. autofunction:: numba.cuda.close

   .. note::
      Compiled functions are associated with the CUDA context.
      This makes it not very useful to close and create new devices, though it
      is certainly useful for choosing which device to use when the machine
      has multiple GPUs.
