
Floating-point pitfalls
=======================

Precision and accuracy
----------------------

For some operations, Numba may use a different algorithm than Python or
Numpy.  The results may not be bit-by-bit compatible.  The difference
should generally be small and within reasonable expectations.  However,
small accumulated differences might produce large differences at the end,
especially if a divergent function is involved.

Linear algebra
''''''''''''''

Numpy forces some linear algebra operations to run in double-precision mode
even when a ``float32`` input is given.  Numba will always observe
the input's precision, and invoke single-precision linear algebra routines
when all inputs are ``float32`` or ``complex64``.


Warnings and errors
-------------------

When calling a :term:`ufunc` created with :func:`~numba.vectorize`,
Numpy will determine whether an error occurred by examining the FPU
error word.  It may then print out a warning or raise an exception
(such as ``RuntimeWarning: divide by zero encountered``),
depending on the current error handling settings.

Depending on how LLVM optimized the ufunc's code, however, some spurious
warnings or errors may appear.  If you get caught by this issue, we
recommend you call :func:`numpy.seterr` to change Numpy's error handling
settings, or the :class:`numpy.errstate` context manager to switch them
temporarily::

   with np.errstate(all='ignore'):
       x = my_ufunc(y)

