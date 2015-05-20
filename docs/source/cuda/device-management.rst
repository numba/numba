
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


.. function:: numba.cuda.select_device(device_id)
   :noindex:

   Create a new CUDA context for the selected *device_id*.  *device_id*
   should be the number of the device (starting from 0; the device order
   is determined by the CUDA libraries).  The context is associated with
   the current thread.  Numba currently allows only one context per thread.

   If successful, this function returns a device instance.

   .. XXX document device instances?


.. function:: numba.cuda.close
   :noindex:

   Explicitly close all contexts in the current thread.

   .. note::
      Compiled functions are associated with the CUDA context.
      This makes it not very useful to close and create new devices, though it
      is certainly useful for choosing which device to use when the machine
      has multiple GPUs.

The Device List
===============

The Device List is a list of all the GPUs in the system, and can be indexed to
obtain a context manager that ensures execution on the selected GPU.

.. attribute:: numba.cuda.gpus
   :noindex:
.. attribute:: numba.cuda.cudadrv.devices.gpus

:py:data:`.gpus` is an instance of the :class:`_DeviceList` class, from which
the current GPU context can also be retrieved:

.. autoclass:: numba.cuda.cudadrv.devices._DeviceList
    :members: current
    :noindex:

