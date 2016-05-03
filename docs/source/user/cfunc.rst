.. _cfunc:

====================================
Creating C callbacks with ``@cfunc``
====================================

Interfacing with some native libraries (for example written in C or C++)
can necessitate writing native callbacks to provide business logic to the
library.  The :func:`numba.cfunc` decorator creates a compiled function
callable from foreign C code, using the signature of your choice.


Basic usage
===========

The ``@cfunc`` decorator has a similar usage to ``@jit``, but with an
important difference: passing a single signature is mandatory.
It determines the visible signature of the C callback::

   from numba import cfunc

   @cfunc("float64(float64, float64)")
   def add(x, y):
       return x + y


The C function object exposes the address of the compiled C callback as
the :attr:`~CFunc.address` attribute, so that you can pass it to any
foreign C or C++ library.  It also exposes a :mod:`ctypes` callback
object pointing to that callback; that object is also callable from
Python, making it easy to check the compiled code::

   @cfunc("float64(float64, float64)")
   def add(x, y):
       return x + y

   print(add.ctypes(4.0, 5.0))  # prints "9.0"


Example
=======

In this example, we are going to be using the ``scipy.integrate.quad``
function.  That function accepts either a regular Python callback or
a C callback wrapped in a :mod:`ctypes` callback object.

Let's define a pure Python integrand and compile it as a
C callback::

   >>> import numpy as np
   >>> from numba import cfunc
   >>> def integrand(t):
           return np.exp(-t) / t**2
      ...:
   >>> nb_integrand = cfunc("float64(float64)")(integrand)

We can pass the ``nb_integrand`` object's :mod:`ctypes` callback to
``scipy.integrate.quad`` and check that the results are the same as with
the pure Python function::

   >>> import scipy.integrate as si
   >>> def do_integrate(func):
           """
           Integrate the given function from 1.0 to +inf.
           """
           return si.quad(func, 1, np.inf)
      ...:
   >>> do_integrate(integrand)
   (0.14849550677592208, 3.8736750296130505e-10)
   >>> do_integrate(nb_integrand.ctypes)
   (0.14849550677592208, 3.8736750296130505e-10)


Using the compiled callback, the integration function does not invoke the
Python interpreter each time it evaluates the integrand.  In our case, the
integration is made 18 times faster::

   >>> %timeit do_integrate(integrand)
   1000 loops, best of 3: 242 µs per loop
   >>> %timeit do_integrate(nb_integrand.ctypes)
   100000 loops, best of 3: 13.5 µs per loop


Signature specification
=======================

The explicit ``@cfunc`` signature can use any Numba types, but only a subset
of them make sense for a C callback.  You should generally limit yourself
to scalar types (such as ``int8`` or ``float64``) or pointers to them
(for example ``types.CPointer(types.int8)``).


Compilation options
===================

A number of keyword-only arguments can be passed to the ``@cfunc``
decorator: ``nopython`` and ``cache``.  Their meaning is similar to those
in the ``@jit`` decorator.
