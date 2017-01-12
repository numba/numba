
=========
Utilities
=========

Dealing with pointers
=====================

These functions can be called from pure Python as well as in
:term:`nopython mode`.


.. function:: numba.carray(ptr, shape, dtype=None)

   Return a Numpy array view over the data pointed to by *ptr* with the
   given *shape*, in C order.  If *dtype* is given, it is used as the array's
   dtype, otherwise the array's dtype is inferred from *ptr*'s type.
   As the returned array is a view, not a copy, writing to it will modify
   the original data.

   *ptr* should be a ctypes pointer object (either a typed pointer
   as created using :func:`~ctypes.POINTER`, or a :class:`~ctypes.c_void_p`).

   *shape* should be an integer or a tuple of integers.

   *dtype* should be a Numpy dtype or scalar class (i.e. both
   ``np.dtype('int8')`` and ``np.int8`` are accepted).


.. function:: numba.farray(ptr, shape, dtype=None)

   Same as :func:`~numba.carray`, but the data is assumed to be laid out
   in Fortran order, and the array view is constructed accordingly.

