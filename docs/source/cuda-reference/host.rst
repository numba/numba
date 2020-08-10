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


Compilation
-----------

Numba provides an entry point for compiling a Python function to PTX without
invoking any of the driver API. This can be useful for:

- Generating PTX that is to be inlined into other PTX code (e.g. from outside
  the Numba / Python ecosystem).
- Generating code when there is no device present.
- Generating code prior to a fork without initializing CUDA.

.. note:: It is the user's responsibility to manage any ABI issues arising from
   the use of compilation to PTX.

.. autofunction:: numba.cuda.compile_ptx


The environment variable ``NUMBA_CUDA_DEFAULT_PTX_CC`` can be set to control
the default compute capability targeted by ``compile_ptx`` - see
:ref:`numba-envvars-gpu-support`. If PTX for the compute capability of the
current device is required, the ``compile_ptx_for_current_device`` function can
be used:

.. autofunction:: numba.cuda.compile_ptx_for_current_device



Measurement
-----------

.. _cuda-profiling:

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


.. _events:

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


.. _streams:

Stream Management
-----------------

Streams allow concurrency of execution on a single device within a given
context. Queued work items in the same stream execute sequentially, but work
items in different streams may execute concurrently. Most operations involving a
CUDA device can be performed asynchronously using streams, including data
transfers and kernel execution. For further details on streams, see the `CUDA C
Programming Guide Streams section
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams>`_.

Streams are instances of :class:`numba.cuda.cudadrv.driver.Stream`:

.. autoclass:: numba.cuda.cudadrv.driver.Stream
   :members: synchronize, auto_synchronize, add_callback, async_done

To create a new stream:

.. autofunction:: numba.cuda.stream

To get the default stream:

.. autofunction:: numba.cuda.default_stream

To get the default stream with an explicit choice of whether it is the legacy
or per-thread default stream:

.. autofunction:: numba.cuda.legacy_default_stream

.. autofunction:: numba.cuda.per_thread_default_stream

To construct a Numba ``Stream`` object using a stream allocated elsewhere, the
``external_stream`` function is provided. Note that the lifetime of external
streams must be managed by the user - Numba will not deallocate an external
stream, and the stream must remain valid whilst the Numba ``Stream`` object is
in use.

.. autofunction:: numba.cuda.external_stream


Runtime
-------

Numba generally uses the Driver API, but it provides a simple wrapper to the
Runtime API so that the version of the runtime in use can be queried. This is
accessed through ``cuda.runtime``, which is an instance of the
:class:`numba.cuda.cudadrv.runtime.Runtime` class:

.. autoclass:: numba.cuda.cudadrv.runtime.Runtime
   :members: get_version
