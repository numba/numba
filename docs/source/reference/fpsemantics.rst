
Floating-point pitfalls
=======================

Precision and accuracy
----------------------

For some operations, Numba may use a different algorithm than Python or
Numpy.  The results may not be bit-by-bit compatible.  The difference
should generally be small and within reasonable expectations.  However,
small accumulated differences might produce large differences at the end,
especially if a divergent function is involved.

Math library implementations
''''''''''''''''''''''''''''

Numba supports a variety of platforms and operating systems, each of which
has its own math library implementation (referred to as ``libm`` from here
in).  The majority of math functions included in ``libm`` have specific
requirements as set out by the IEEE 754 standard (like ``sin()``, ``exp()``
etc.), but each implementation may have bugs.  Thus, on some platforms
Numba has to exercise special care in order to workaround known ``libm``
issues.

Another typical problem is when an operating system's ``libm`` function
set is incomplete and needs to be supplemented by additional functions.
These are provided with reference to the IEEE 754 and C99 standards
and are often implemented in Numba in a manner similar to equivalent
CPython functions.

In particular, math library issues are known to affect Python 2.7 builds
on Windows, since Python 2.7 requires the use of an obsolete version of
the Microsoft Visual Studio compiler.

Linear algebra
''''''''''''''

Numpy forces some linear algebra operations to run in double-precision mode
even when a ``float32`` input is given.  Numba will always observe
the input's precision, and invoke single-precision linear algebra routines
when all inputs are ``float32`` or ``complex64``.

The implementations of the ``numpy.linalg`` routines in Numba only support the
floating point types that are used in the LAPACK functions that provide
the underlying core functionality. As a result  only ``float32``, ``float64``,
``complex64`` and ``complex128`` types are supported. If a user has e.g. an
``int32`` type, an appropriate type conversion must be performed to a
floating point type prior to its use in these routines. The reason for this
decision is to essentially avoid having to replicate type conversion choices
made in Numpy and to also encourage the user to choose the optimal floating
point type for the operation they are undertaking.


Mixed-types operations
''''''''''''''''''''''

Numpy will most often return a ``float64`` as a result of a computation
with mixed integer and floating-point operands (a typical example is the
power operator ``**``).  Numba by contrast will select the highest precision
amongst the floating-point operands, so for example ``float32 ** int32``
will return a ``float32``, regardless of the input values.  This makes
performance characteristics easier to predict, but you should explicitly
cast the input to ``float64`` if you need the extra precision.


.. _ufunc-fpu-errors:

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

