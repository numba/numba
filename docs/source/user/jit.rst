.. _jit:

===================================
Compiling Python code with ``@jit``
===================================

Numba provides several utilities for code generation, but its central
feature is the :func:`numba.jit` decorator.  Using this decorator, you can mark
a function for optimization by Numba's JIT compiler.  Various invocation
modes trigger differing compilation options and behaviours.


Basic usage
===========

.. _jit-lazy:

Lazy compilation
----------------

The recommended way to use the ``@jit`` decorator is to let Numba decide
when and how to optimize::

   from numba import jit

   @jit
   def f(x, y):
       # A somewhat trivial example
       return x + y

In this mode, compilation will be deferred until the first function
execution.  Numba will infer the argument types at call time, and generate
optimized code based on this information.  Numba will also be able to
compile separate specializations depending on the input types.  For example,
calling the ``f()`` function above with integer or complex numbers will
generate different code paths::

   >>> f(1, 2)
   3
   >>> f(1j, 2)
   (2+1j)

Eager compilation
-----------------

You can also tell Numba the function signature you are expecting.  The
function ``f()`` would now look like::

   from numba import jit, int32

   @jit(int32(int32, int32))
   def f(x, y):
       # A somewhat trivial example
       return x + y

``int32(int32, int32)`` is the function's signature.  In this case, the
corresponding specialization will be compiled by the ``@jit`` decorator,
and no other specialization will be allowed. This is useful if you want
fine-grained control over types chosen by the compiler (for example,
to use single-precision floats).

If you omit the return type, e.g. by writing ``(int32, int32)`` instead of
``int32(int32, int32)``, Numba will try to infer it for you.  Function
signatures can also be strings, and you can pass several of them as a list;
see the :func:`numba.jit` documentation for more details.

Of course, the compiled function gives the expected results::

   >>> f(1,2)
   3

and if we specified ``int32`` as return type, the higher-order bits get
discarded::

   >>> f(2**31, 2**31 + 1)
   1


Calling and inlining other functions
====================================

Numba-compiled functions can call other compiled functions.  The function
calls may even be inlined in the native code, depending on optimizer
heuristics.  For example::

   @jit
   def square(x):
       return x ** 2

   @jit
   def hypot(x, y):
       return math.sqrt(square(x) + square(y))

The ``@jit`` decorator *must* be added to any such library function,
otherwise Numba may generate much slower code.


Signature specifications
========================

Explicit ``@jit`` signatures can use a number of types.  Here are some
common ones:

* ``void`` is the return type of functions returning nothing (which
  actually return :const:`None` when called from Python)
* ``intp`` and ``uintp`` are pointer-sized integers (signed and unsigned,
  respectively)
* ``intc`` and ``uintc`` are equivalent to C ``int`` and ``unsigned int``
  integer types
* ``int8``, ``uint8``, ``int16``, ``uint16``, ``int32``, ``uint32``,
  ``int64``, ``uint64`` are fixed-width integers of the corresponding bit
  width (signed and unsigned)
* ``float32`` and ``float64`` are single- and double-precision floating-point
  numbers, respectively
* ``complex64`` and ``complex128`` are single- and double-precision complex
  numbers, respectively
* array types can be specified by indexing any numeric type, e.g. ``float32[:]``
  for a one-dimensional single-precision array or ``int8[:,:]`` for a
  two-dimensional array of 8-bit integers.


Compilation options
===================

A number of keyword-only arguments can be passed to the ``@jit`` decorator.

.. _jit-nopython:

``nopython``
------------

Numba has two compilation modes: :term:`nopython mode` and
:term:`object mode`.  The former produces much faster code, but has
limitations that can force Numba to fall back to the latter.  To prevent
Numba from falling back, and instead raise an error, pass ``nopython=True``.

::

   @jit(nopython=True)
   def f(x, y):
       return x + y

.. seealso:: :ref:`troubleshooting`

.. _jit-nogil:

``nogil``
---------

Whenever Numba optimizes Python code to native code that only works on
native types and variables (rather than Python objects), it is not necessary
anymore to hold Python's :py:term:`global interpreter lock` (GIL).
Numba will release the GIL when entering such a compiled function if you
passed ``nogil=True``.

::

   @jit(nogil=True)
   def f(x, y):
       return x + y

Code running with the GIL released runs concurrently with other
threads executing Python or Numba code (either the same compiled function,
or another one), allowing you to take advantage of multi-core systems.
This will not be possible if the function is compiled in :term:`object mode`.

When using ``nogil=True``, you'll have to be wary of the usual pitfalls
of multi-threaded programming (consistency, synchronization, race conditions,
etc.).

.. _jit-cache:

``cache``
---------

To avoid compilation times each time you invoke a Python program,
you can instruct Numba to write the result of function compilation into
a file-based cache.  This is done by passing ``cache=True``::

   @jit(cache=True)
   def f(x, y):
       return x + y
