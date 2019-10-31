.. _cuda-array-interface:

====================
CUDA Array Interface
====================

The *cuda array inteface* is created for interoperability between different
implementation of GPU array-like objects in various projects.  The idea is
borrowed from the `numpy array interface`_.


.. note::
    Currently, we only define the Python-side interface.  In the future, we may
    add a C-side interface for efficient exchange of the information in
    compiled code.


Python Interface Specification
==============================

.. note:: Experimental feature.  Specification may change.

The ``__cuda_array_interface__`` attribute returns a dictionary that must
contain the following entries:

- **shape**: ``(integer, ...)``

    A tuple of `int` (or `long`) representing the size of each dimension.

- **typestr**: `str`

    The type string.  This has the same definition as *typestr* in the
    `numpy array interface`_.

- **data**: `(integer, boolean)`

    The **data** is a 2-tuple.  The first element is the data pointer
    as a Python `int` (or `long`).  The data must be device-accessible.
    For zero-size arrays, use `0` here.
    The second element is the read-only flag as a Python `bool`.

    Because the user of the interface may or may not be in the same context,
    the most common case is to use ``cuPointerGetAttribute`` with
    ``CU_POINTER_ATTRIBUTE_DEVICE_POINTER`` in the CUDA driver API (or the
    equivalent CUDA Runtime API) to retrieve a device pointer that
    is usable in the currently active context.

- **version**: `integer`

    An integer for the version of the interface being exported.
    The current version is *2*.


The following are optional entries:

- **strides**: ``None`` or ``(integer, ...)``

    If **strides** is not given, or it is ``None``, the array is in
    C-contiguous layout. Otherwise, a tuple of `int` (or `long`) is explicitly
    given for representing the number of bytes to skip to access the next
    element at each dimension.

- **descr**

    This is for describing more complicated types.  This follows the same
    specification as in the `numpy array interface`_.

- **mask**: ``None`` or object exposing the ``__cuda_array_interface__``

    If ``None`` then all values in **data** are valid. All elements of the mask
    array should be interpreted only as true or not true indicating which
    elements of this array are valid. This has the same definition as *mask*
    in the `numpy array interface`_.

    .. note:: Numba does not currently support working with masked CUDA arrays
              and will raise a `NotImplementedError` exception if one is passed
              to a GPU function.




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
