.. _cuda-array-interface:

================================
CUDA Array Interface (Version 3)
================================

The *CUDA Array Interface* (or CAI) is created for interoperability between
different implementations of GPU array-like objects in various projects. The
idea is borrowed from the `NumPy array interface`_.


.. note::
    Currently, we only define the Python-side interface.  In the future, we may
    add a C-side interface for efficient exchange of the information in
    compiled code.


Python Interface Specification
==============================

.. note:: Experimental feature.  Specification may change.

The ``__cuda_array_interface__`` attribute returns a dictionary (``dict``)
that must contain the following entries:

- **shape**: ``(integer, ...)``

  A tuple of ``int`` (or ``long``) representing the size of each dimension.

- **typestr**: ``str``

  The type string.  This has the same definition as ``typestr`` in the
  `numpy array interface`_.

- **data**: ``(integer, boolean)``

  The **data** is a 2-tuple.  The first element is the data pointer
  as a Python ``int`` (or ``long``).  The data must be device-accessible.
  For zero-size arrays, use ``0`` here.
  The second element is the read-only flag as a Python ``bool``.

  Because the user of the interface may or may not be in the same context,
  the most common case is to use ``cuPointerGetAttribute`` with
  ``CU_POINTER_ATTRIBUTE_DEVICE_POINTER`` in the CUDA driver API (or the
  equivalent CUDA Runtime API) to retrieve a device pointer that
  is usable in the currently active context.

- **version**: ``integer``

  An integer for the version of the interface being exported.
  The current version is *3*.


The following are optional entries:

- **strides**: ``None`` or ``(integer, ...)``

  If **strides** is not given, or it is ``None``, the array is in
  C-contiguous layout. Otherwise, a tuple of ``int`` (or ``long``) is explicitly
  given for representing the number of bytes to skip to access the next
  element at each dimension.

- **descr**

  This is for describing more complicated types.  This follows the same
  specification as in the `numpy array interface`_.

- **mask**: ``None`` or object exposing the ``__cuda_array_interface__``

  If ``None`` then all values in **data** are valid. All elements of the mask
  array should be interpreted only as true or not true indicating which
  elements of this array are valid. This has the same definition as ``mask``
  in the `numpy array interface`_.

  .. note:: Numba does not currently support working with masked CUDA arrays
            and will raise a ``NotImplementedError`` exception if one is passed
            to a GPU function.

- **stream**: ``None`` or ``integer``

  An optional stream upon which synchronization must take place at the point of
  consumption, either by synchronizing on the stream or enqueuing operations on
  the data on the given stream. Integer values in this entry are as follows:

  - ``0``: This is disallowed as it would be ambiguous between ``None`` and the
    default stream, and also between the legacy and per-thread default streams.
    Any use case where ``0`` might be given should either use ``None``, ``1``,
    or ``2`` instead for clarity.
  - ``1``: The legacy default stream.
  - ``2``: The per-thread default stream.
  - Any other integer: a ``cudaStream_t`` represented as a Python integer.

  See the
  :ref:`cuda-array-interface-synchronization` section below for further details.

  In a future revision of the interface, this entry may be expanded (or another
  entry added) so that an event to synchronize on can be specified instead of a
  stream.


.. _cuda-array-interface-synchronization:

Synchronization
---------------

Definitions
~~~~~~~~~~~

When discussing synchronization, the following definitions are used:

- *Producer*: The library / object on which ``__cuda_array_interface__`` is
  accessed.
- *Consumer*: The library / function that accesses the
  ``__cuda_array_interface__`` of the Producer.
- *User Code*: Code that induces a Producer and Consumer to share data through
  the CAI.
- *User*: The person writing or maintaining the User Code. The User may
  implement User Code without knowledge of the CAI, since the CAI accesses can
  be  hidden from their view.

In the following example:

.. code-block:: python

   import cupy
   from numba import cuda

   @cuda.jit
   def add(x, y, out):
       start = cuda.grid(1)
       stride = cuda.gridsize(1)
       for i in range(start, x.shape[0], stride):
           out[i] = x[i] + y[i]

   a = cupy.arange(10)
   b = a * 2
   out = cupy.zeros_like(a)

   add[1, 32](a, b, out)

When the ``add`` kernel is launched:

- ``a``, ``b``, ``out`` are Producers.
- The ``add`` kernel is the Consumer.
- The User Code is specifically ``add[1, 32](a, b, out)``.
- The author of the code is the User.


Design Motivations
~~~~~~~~~~~~~~~~~~

Elements of the CAI design related to synchronization seek to fulfill these
requirements:

1. Producers and Consumers that exchange data through the CAI must be able to do
   so without data races.
2. Requirement 1 should be met without requiring the user to be
   aware of any particulars of the CAI - in other words, exchanging data between
   Producers and Consumers that operate on data asynchronously should be correct
   by default.

   - An exception to this requirement is made for Producers and Consumers that
     explicitly document that the User is required to take additional steps to
     ensure correctness with respect to synchronization. In this case, Users
     are required to understand the details of the CUDA Array Interface, and
     the Producer/Consumer library documentation must specify the steps that
     Users are required to take.

     Use of this exception should be avoided where possible, as it is provided
     for libraries that cannot implement the synchronization semantics without
     the involvement of the User - for example, those interfacing with
     third-party libraries oblivious to the CUDA Array Interface.

3. Where the User is aware of the particulars of the CAI and implementation
   details of the Producer and Consumer, they should be able to, at their
   discretion, override some of the synchronization semantics of the interface
   to reduce the synchronization overhead. Overriding synchronization semantics
   implies that:

   - The CAI design, and the design and implementation of the Producer and
     Consumer do not specify or guarantee correctness with respect to data
     races.
   - Instead, the User is responsible for ensuring correctness with respect to
     data races.


Interface Requirements
~~~~~~~~~~~~~~~~~~~~~~

The ``stream`` entry enables Producers and Consumers to avoid hazards when
exchanging data. Expected behaviour of the Consumer is as follows:

* When ``stream`` is not present or is ``None``:

  - No synchronization is required on the part of the Consumer.
  - The Consumer may enqueue operations on the underlying data immediately on
    any stream.

* When ``stream`` is an integer, its value indicates the stream on which the
  Producer may have in-progress operations on the data, and which the Consumer
  is expected to either:

  - Synchronize on before accessing the data, or
  - Enqueue operations in when accessing the data.

  The Consumer can choose which mechanism to use, with the following
  considerations:

  - If the Consumer synchronizes on the provided stream prior to accessing the
    data, then it must ensure that no computation can take place in the provided
    stream until its operations in its own choice of stream have taken place.
    This could be achieved by either:

    - Placing a wait on an event in the provided stream that occurs once all
      of the Consumer's operations on the data are completed, or
    - Avoiding returning control to the user code until after its operations
      on its own stream have completed.

  - If the consumer chooses to only enqueue operations on the data in the
    provided stream, then it may return control to the User code immediately
    after enqueueing its work, as the work will all be serialized on the
    exported array's stream. This is sufficient to ensure correctness even if
    the User code were to induce the Producer to subsequently start enqueueing
    more work on the same stream.

* If the User has set the Consumer to ignore CAI synchronization semantics, the
  Consumer may assume it can operate on the data immediately in any stream with
  no further synchronization, even if the ``stream`` member has an integer
  value.


When exporting an array through the CAI, Producers must ensure that:

* If there is work on the data enqueued in one or more streams, then
  synchronization on the provided ``stream`` is sufficient to ensure
  synchronization with all pending work.

  - If the Producer has no enqueued work, or work only enqueued on the stream
    identified by ``stream``, then this condition is met.
  - If the Producer has enqueued work on the data on multiple streams, then it
    must enqueue events on those streams that follow the enqueued work, and
    then wait on those events in the provided ``stream``. For example:

    1. Work is enqueued by the Producer on streams ``7``, ``9``, and ``15``.
    2. Events are then enqueued on each of streams ``7``, ``9``, and ``15``.
    3. Producer then tells stream ``3`` to wait on the events from Step 2, and
       the ``stream`` entry is set to ``3``.

* If there is no work enqueued on the data, then the ``stream`` entry may be
  either ``None``, or not provided.

Optionally, to facilitate the User to relax the conformance to synchronization
semantics:

* Producers may provide a configuration option to always set ``stream`` to
  ``None``.
* Consumers may provide a configuration option to ignore the value of ``stream``
  and act as if it were ``None`` or not provided.  This elides synchronization
  on the Producer-provided streams, and allows enqueuing work on streams other
  than that provided by the Producer.

These options should not be set by default in either a Producer or a Consumer.
The exact mechanism by which these options are set, and related options that
Producers or Consumers might provide to allow the user further control over
synchronization behavior are not prescribed by the CAI specification.


Synchronization in Numba
~~~~~~~~~~~~~~~~~~~~~~~~

Numba is neither strictly a Producer nor a Consumer - it may be used to
implement either by a User. In order to facilitate the correct implementation of
synchronization semantics, Numba exhibits the following behaviors related to
synchronization of the interface:

- When Numba acts as a Consumer (for example when an array-like object is passed
  to a kernel launch): If ``stream`` is an integer, then Numba will immediately
  synchronize on the provided ``stream``. A Numba Device Array created from an
  array-like object has its *default stream* set to the provided stream.

- When Numba acts as a Producer (when the ``__cuda_array_interface__`` property
  of a Numba Device Array is accessed): If the exported Device Array has a
  *default stream*, then it is given as the ``stream`` entry. Otherwise,
  ``stream`` is set to ``None``.

.. note:: In Numba's terminology, the *default stream* for a Device Array is a
          property of the Device Array specifying the stream in which Numba will
          enqueue asynchronous transfers if no other stream is provided as an
          argument to the function invoking the transfer. It is not the same as
          the `Default Stream
          <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#default-stream>`_
          in normal CUDA terminology.

Numba's synchronization behavior results in the following intended
consequences:

- Exchanging data either as a Producer or a Consumer will be correct without
  the need for any further action from the User, provided that the other side
  of the interaction also follows the CAI synchronization semantics.
- The User is expected to either:

  - Avoid launching kernels or other operations on streams that
    are not the default stream for their parameters, or
  - When launching operations on a stream that is not the default stream for
    a given parameter, they should then insert an event into the stream that
    they are operating in, and wait on that event in the default stream for
    the parameter. For an example of this, :ref:`see below
    <example-multi-streams>`.

The User may override synchronization behavior in Numba by setting the
environment variable ``NUMBA_CUDA_ARRAY_INTERFACE_SYNC`` or the config variable
``CUDA_ARRAY_INTERFACE_SYNC`` to ``0`` (see :ref:`GPU Support Environment
Variables <numba-envvars-gpu-support>`).  When set, Numba will not synchronize
on the streams of imported arrays, and it is the responsibility of the user to
ensure correctness with respect to stream synchronization. Synchronization when
creating a Numba Device Array from an object exporting the CUDA Array Interface
may also be elided by passing ``sync=False`` when creating the Numba Device
Array with :func:`numba.cuda.as_cuda_array` or
:func:`numba.cuda.from_cuda_array_interface`.

There is scope for Numba's synchronization implementation to be optimized in
the future, by eliding synchronizations when a kernel or driver API operation
(e.g.  a memcopy or memset) is launched on the same stream as an imported
array.


.. _example-multi-streams:

An example launching on an array's non-default stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to ensure that a Consumer can safely consume an array
with a default stream when it is passed to a kernel launched in a different
stream.

First we need to import Numba and a consumer library (a fictitious library named
``other_cai_library`` for this example):

.. code-block:: python

   from numba import cuda, int32, void
   import other_cai_library

Now we'll define a kernel - this initializes the elements of the array, setting
each entry to its index:

.. code-block:: python

   @cuda.jit(void, int32[::1])
   def initialize_array(x):
       i = cuda.grid(1)
       if i < len(x):
           x[i] = i

Next we will create two streams:

.. code-block:: python

   array_stream = cuda.stream()
   kernel_stream = cuda.stream()

Then create an array with one of the streams as its default stream:

.. code-block:: python

   N = 16384
   x = cuda.device_array(N, stream=array_stream)

Now we launch the kernel in the other stream:

.. code-block:: python

   nthreads = 256
   nblocks = N // nthreads

   initialize_array[nthreads, nblocks, kernel_stream](x)

If we were to pass ``x`` to a Consumer now, there is a risk that it may operate on
it in ``array_stream`` whilst the kernel is still running in ``kernel_stream``.
To prevent operations in ``array_stream`` starting before the kernel launch is
finished, we create an event and wait on it:

.. code-block:: python

   # Create event
   evt = cuda.event()
   # Record the event after the kernel launch in kernel_stream
   evt.record(kernel_stream)
   # Wait for the event in array_stream
   evt.wait(array_stream)

It is now safe for ``other_cai_library`` to consume ``x``:

.. code-block:: python

   other_cai_library.consume(x)


Lifetime management
-------------------

Data
~~~~

Obtaining the value of the ``__cuda_array_interface__`` property of any object
has no effect on the lifetime of the object from which it was created. In
particular, note that the interface has no slot for the owner of the data.

The User code must preserve the lifetime of the object owning the data for as
long as the Consumer might user it.


Streams
~~~~~~~

Like data, CUDA streams also have a finite lifetime. It is therefore required
that a Producer exporting data on the interface with an associated stream
ensures that the exported stream's lifetime is equal to or surpasses the
lifetime of the object from which the interface was exported.


Lifetime management in Numba
----------------------------

Producing Arrays
~~~~~~~~~~~~~~~~

Numba takes no steps to maintain the lifetime of an object from which the
interface is exported - it is the user's responsibility to ensure that the
underlying object is kept alive for the duration that the exported interface
might be used.

The lifetime of any Numba-managed stream exported on the interface is guaranteed
to equal or surpass the lifetime of the underlying object, because the
underlying object holds a reference to the stream.

.. note:: Numba-managed streams are those created with
          ``cuda.default_stream()``, ``cuda.legacy_default_stream()``, or
          ``cuda.per_thread_default_stream()``. Streams not managed by Numba
          are created from an external stream with ``cuda.external_stream()``.


Consuming Arrays
~~~~~~~~~~~~~~~~

Numba provides two mechanisms for creating device arrays from objects exporting
the CUDA Array Interface. Which to use depends on whether the created device
array should maintain the life of the object from which it is created:

- ``as_cuda_array``: This creates a device array that holds a reference to the
  owning object. As long as a reference to the device array is held, its
  underlying data will also be kept alive, even if all other references to the
  original owning object have been dropped.
- ``from_cuda_array_interface``: This creates a device array with no reference
  to the owning object by default. The owning object, or some other object to
  be considered the owner can be passed in the ``owner`` parameter.

The interfaces of these functions are:

.. automethod:: numba.cuda.as_cuda_array

.. automethod:: numba.cuda.from_cuda_array_interface


Pointer Attributes
------------------

Additional information about the data pointer can be retrieved using
``cuPointerGetAttribute`` or ``cudaPointerGetAttributes``.  Such information
include:

- the CUDA context that owns the pointer;
- is the pointer host-accessible?
- is the pointer a managed memory?


.. _numpy array interface: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__


Differences with CUDA Array Interface (Version 0)
-------------------------------------------------

Version 0 of the CUDA Array Interface did not have the optional **mask**
attribute to support masked arrays.


Differences with CUDA Array Interface (Version 1)
-------------------------------------------------

Versions 0 and 1 of the CUDA Array Interface neither clarified the
**strides** attribute for C-contiguous arrays nor specified the treatment for
zero-size arrays.


Differences with CUDA Array Interface (Version 2)
-------------------------------------------------

Prior versions of the CUDA Array Interface made no statement about
synchronization.


Interoperability
----------------

The following Python libraries have adopted the CUDA Array Interface:

- Numba
- `CuPy <https://docs-cupy.chainer.org/en/stable/reference/interoperability.html>`_
- `PyTorch <https://pytorch.org>`_
- `PyArrow <https://arrow.apache.org/docs/python/generated/pyarrow.cuda.Context.html#pyarrow.cuda.Context.buffer_from_object>`_
- `mpi4py <https://mpi4py.readthedocs.io/en/latest/overview.html#support-for-cuda-aware-mpi>`_
- `ArrayViews <https://github.com/xnd-project/arrayviews>`_
- `JAX <https://jax.readthedocs.io/en/latest/index.html>`_
- The RAPIDS stack:

    - `cuDF <https://rapidsai.github.io/projects/cudf/en/0.11.0/10min-cudf-cupy.html>`_
    - `cuML <https://docs.rapids.ai/api/cuml/nightly/>`_
    - `cuSignal <https://github.com/rapidsai/cusignal>`_
    - `RMM <https://docs.rapids.ai/api/rmm/stable/>`_

If your project is not on this list, please feel free to report it on the `Numba issue tracker <https://github.com/numba/numba/issues>`_.
