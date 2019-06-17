============================
NBEP 4: Defining C callbacks
============================

:Author: Antoine Pitrou
:Date: April 2016
:Status: Draft


Interfacing with some native libraries (for example written in C
or C++) can necessitate writing native callbacks to provide business logic
to the library.  Some Python-facing libraries may also provide the
alternative of passing a ctypes-wrapped native callback instead of a
Python callback for better performance.  A simple example is the
``scipy.integrate`` package where the user passes the function to be
integrated as a callback.

Users of those libraries may want to benefit from the performance advantage
of running purely native code, while writing their code in Python.
This proposal outlines a scheme to provide such a functionality in
Numba.


Basic usage
===========

We propose adding a new decorator, ``@cfunc``, importable from the main
package.  This decorator allows defining a callback as in the following
example::

   from numba import cfunc
   from numba.types import float64

   # A callback with the C signature `double(double)`

   @cfunc(float64(float64), nopython=True)
   def integrand(x):
       return 1 / x


The ``@cfunc`` decorator returns a "C function" object holding the
resources necessary to run the given compiled function (for example its
LLVM module).  This object has several attributes and methods:

* the ``ctypes`` attribute is a ctypes function object representing
  the native function.

* the ``address`` attribute is the address of the native function code, as
  an integer (note this can also be computed from the ``ctypes`` attribute).

* the ``native_name`` attribute is the symbol under which the function
  can be looked up inside the current process.

* the ``inspect_llvm()`` method returns the IR for the LLVM module
  in which the function is compiled.  It is expected that the ``native_name``
  attribute corresponds to the function's name in the LLVM IR.

The general signature of the decorator is ``cfunc(signature, **options)``.

The ``signature`` must specify the argument types and return type of the
function using Numba types.  In contrary to ``@jit``, the return type cannot
be omitted.

The ``options`` are keyword-only parameters specifying compilation options.
We are expecting that the standard ``@jit`` options (``nopython``,
``forceobj``, ``cache``) can be made to work with ``@cfunc``.


Calling from Numba-compiled functions
-------------------------------------

While the intended use is to pass a callback's address to foreign C
code expecting a function pointer, it should be made possible to call
the C callback from a Numba-compiled function.


Passing array data
==================

Native platform ABIs as used by C or C++ don't have the notion of a shaped
array as in Numpy.  One common solution is to pass a raw data pointer and
one or several size arguments (depending on dimensionality).  Numba must
provide a way to rebuild an array view of this data inside the callback.

::

   from numba import cfunc, carray
   from numba.types import float64, CPointer, void, intp

   # A callback with the C signature `void(double *, double *, size_t)`

   @cfunc(void(CPointer(float64), CPointer(float64), intp))
   def invert(in_ptr, out_ptr, n):
       in_ = carray(in_ptr, (n,))
       out = carray(out_ptr, (n,))
       for i in range(n):
           out[i] = 1 / in_[i]


The ``carray`` function takes ``(pointer, shape, dtype)`` arguments
(``dtype`` being optional) and returns a C-layout array view over the
data *pointer*, with the given *shape* and *dtype*.  *pointer* must
be a ctypes pointer object (not a Python integer).  The array's
dimensionality corresponds to the *shape* tuple's length.  If *dtype*
is not given, the array's dtype corresponds to the *pointer*'s pointee
type.

The ``farray`` function is similar except that it returns a F-layout
array view.


Error handling
==============

There is no standard mechanism in C for error reporting.  Unfortunately,
Numba currently doesn't handle ``try..except`` blocks, which makes it more
difficult for the user to implement the required error reporting scheme.
The current stance of this proposal is to let users guard against invalid
arguments where necessary, and do whatever is required to inform the caller
of the error.

Based on user feedback, we can later add support for some error reporting
schemes, such as returning an integer error code depending on whether an
exception was raised, or setting ``errno``.


Deferred topics
===============

Ahead-of-Time compilation
-------------------------

This proposal doesn't make any provision for AOT compilation of C callbacks.
It would probably necessitate a separate API (a new method on the
``numba.pycc.CC`` object), and the implementation would require exposing
a subset of the C function object's functionality from the compiled C
extension module.

Opaque data pointers
--------------------

Some libraries allow passing an opaque data pointer (``void *``) to a
user-provided callback, to provide any required context for execution
of the callback.  Taking advantage of this functionality would require
adding specific support in Numba, for example the ability to do generic
conversion from ``types.voidptr`` and to take the address of a
Python-facing ``jitclass`` instance.
