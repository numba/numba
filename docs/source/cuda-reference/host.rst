CUDA Host API
=============

Device Management
-----------------

Device detection and enquiry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following functions are available for querying the available hardware:

.. autofunction:: numba.cuda.is_available

.. autofunction:: numba.cuda.detect

Context management
~~~~~~~~~~~~~~~~~~

CUDA Python functions execute within a CUDA context. Each CUDA device in a
system has an associated CUDA context, and Numba presently allows only one context
per thread. For further details on CUDA Contexts, refer to the `CUDA Driver API
Documentation on Context Management
<http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html>`_ and the
`CUDA C Programming Guide Context Documentation
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#context>`_. CUDA Contexts
are instances of the :class:`~numba.cuda.cudadrv.driver.Context` class:

.. autoclass:: numba.cuda.cudadrv.driver.Context
   :members: reset, get_memory_info, push, pop

The following functions can be used to get or select the context:

.. autofunction:: numba.cuda.current_context
.. autofunction:: numba.cuda.require_context

The following functions affect the current context:

.. autofunction:: numba.cuda.synchronize
.. autofunction:: numba.cuda.close

Device management
~~~~~~~~~~~~~~~~~

Numba maintains a list of supported CUDA-capable devices:

.. attribute:: numba.cuda.gpus

   An indexable list of supported CUDA devices. This list is indexed by integer
   device ID.

Alternatively, the current device can be obtained:

.. function:: numba.cuda.gpus.current

   Return the currently-selected device.

Getting a device through :attr:`numba.cuda.gpus` always provides an instance of
:class:`numba.cuda.cudadrv.devices._DeviceContextManager`, which acts as a
context manager for the selected device:

.. autoclass:: numba.cuda.cudadrv.devices._DeviceContextManager

One may also select a context and device or get the current device using the
following three functions:

.. autofunction:: numba.cuda.select_device
.. autofunction:: numba.cuda.get_current_device
.. autofunction:: numba.cuda.list_devices

The :class:`numba.cuda.cudadrv.driver.Device` class can be used to enquire about
the functionality of the selected device:

.. class:: numba.cuda.cudadrv.driver.Device

   The device associated with a particular context.

   .. attribute:: compute_capability

      A tuple, *(major, minor)* indicating the supported compute capability.

   .. attribute:: id

      The integer ID of the device.

   .. attribute:: name

      The name of the device (e.g. "GeForce GTX 970")

   .. method:: reset

      Delete the context for the device. This will destroy all memory
      allocations, events, and streams created within the context.

Measurement
-----------

Profiling
~~~~~~~~~

The NVidia Visual Profiler can be used directly on executing CUDA Python code -
it is not a requirement to insert calls to these functions into user code.
However, these functions can be used to allow profiling to be performed
selectively on specific portions of the code. For further information on
profiling, see the `NVidia Profiler User's Guide
<docs.nvidia.com/cuda/profiler-users-guide/>`_.

.. autofunction:: numba.cuda.profile_start
.. autofunction:: numba.cuda.profile_stop
.. autofunction:: numba.cuda.profiling

Events
~~~~~~

Events can be used to monitor the progress of execution and to record the
timestamps of specific points being reached. Event creation returns immediately,
and the created event can be queried to determine if it has been reached. For
further information, see the `CUDA C Programming Guide Events section
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#events>`_.

The following functions are used for creating and measuring the time between
events:

.. autofunction:: numba.cuda.event
.. autofunction:: numba.cuda.event_elapsed_time

Events are instances of the :class:`numba.cuda.cudadrv.driver.Event` class:

.. autoclass:: numba.cuda.cudadrv.driver.Event
   :members: query, record, synchronize, wait

Stream Management
-----------------

Streams allow concurrency of execution on a single device within a given
context. Queued work items in the same stream execute sequentially, but work
items in different streams may execute concurrently. Most operations involving a
CUDA device can be performed asynchronously using streams, including data
transfers and kernel execution. For further details on streams, see the `CUDA C
Programming Guide Streams section
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams>`_.

To create a stream:

.. autofunction:: numba.cuda.stream

Streams are instances of :class:`numba.cuda.cudadrv.driver.Stream`:

.. autoclass:: numba.cuda.cudadrv.driver.Stream
   :members: synchronize, auto_synchronize

