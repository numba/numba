.. _cuda-array-interface:

================================
CUDA Array Interface (Version 3)
================================

The *cuda array interface* is created for interoperability between different
implementation of GPU array-like objects in various projects.  The idea is
borrowed from the `numpy array interface`_.


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
  elements of this array are valid. This has the same definition as *mask* in
  the `numpy array interface`_.

  .. note:: Numba does not currently support working with masked CUDA arrays
            and will raise a `NotImplementedError` exception if one is passed
            to a GPU function.


Synchronization
---------------

For its implicit synchronization, the consumer is expected to operate on the
data on the default stream, or synchronized with an event on the default stream.
There are various mechanisms that could be used for this synchronization
depending on the environment in which the framework(s) exchanging data using the
interface are in. These include:

- Using ``cudaStreamWaitEvent`` in the runtime API,
- Using ``cuStreamWaitEvent`` in the Driver API,
- Using :func:`numba.cuda.cudadrv.driver.Event.wait` in Numba,
- or, some other similar mechanism provided by another framework.

Conversely, the producer is expected to synchronize the default stream with any
operations on other streams that could be operating on the data at the time of
export. This can be achieved by recording an event in the non-default
stream that is operating on the data, and waiting on it in the default stream.
If there are multiple streams operating on the data, then one event per
non-default stream could be recorded, and all waited on in the default stream.

There can exist a scenario in which the consumer knows that no synchronization
is required - for example, if they are using the same stream across two
different frameworks. In this case they may choose to avoid waiting on an event
in the default stream.


Synchronization in Numba
------------------------

Numba is neither strictly a producer nor a consumer - it may be used to
implement either by a user. In order to facilitate the correct implementation of
sychronization semantics, Numba exhibits the following behaviors related to
synchronization of the interface:

- If a device array is bound to a stream, then when the interface is exported
  (i.e. at the time the ``__cuda_array_interface__`` property of a device array
  is accessed), Numba will record an event in the array's stream and insert a
  wait on the event in the default stream.
- If a device array is imported (e.g. using :func:`cuda.as_cuda_array` on an
  object providing ``__cuda_array_interface__``, then Numba will bind the
  imported array to the default stream so that any operations on it will be
  synchronized correctly even if the user does nothing to bind the array to a
  stream.
- If the synchronization on the default stream when importing is undesired (e.g.
  in the scenario described above where no synchronization is necessary) then
  the binding of the array to the default stream can be avoided by passing
  ``sync=False`` to either of :func:`cuda.from_cuda_array_interface` or
  :func:`as_cuda_array`.

This means that:

- If the user is only operating on the default stream, then no further action on
  the part of the user is required.
- If the user performs operations on non-default streams, then device arrays
  they use that may be exported to other frameworks should be operated on in the
  stream that they are bound to - this binding can take place either at
  construction time, or by using the
  :func:`numba.cuda.cudadrv.devicearray.DeviceNDArray.bind` method.
- If synchronization is to be avoided on both import and export, then device
  arrays should not be bound to a particular stream (instead the stream must be
  specified for operations on the arrays, such as kernel launches and data
  transfers), and ``sync=False`` should be passed to functions importing arrays
  through the interface.


Lifetime management
-------------------

Obtaining the value of the ``__cuda_array_interface__`` property of any object
has no effect on the lifetime of the object from which it was created. In
particular, note that the interface has no slot for the owner of the data.

It is therefore imperative for a consumer to retain a reference to the object
owning the data for as long as they make use of the data.


Lifetime management in Numba
----------------------------

Numba provides two mechanisms for creating device arrays. Which to use depends
on whether the created device array should maintain the life of the object from
which it is created:

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
synchronization or lifetime management.


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
