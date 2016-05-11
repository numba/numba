
Deviations from Python semantics
================================

Integer width
-------------

While Python has arbitrary-sized integers, integers in Numba-compiled
functions get a fixed size (either through :term:`type inference`, or
from an explicit specification by the user).  This means that arithmetic
operations can wrapround or produce undefined results or overflow.

Boolean inversion
-----------------

Calling the bitwise complement operator (the ``~`` operator) on a Python
boolean returns an integer, while the same operator on a Numpy boolean
returns another boolean::

   >>> ~True
   -2
   >>> ~np.bool_(True)
   False

Numba follows the Numpy semantics.

Global and closure variables
----------------------------

In :term:`nopython mode`, global and closure variables are *frozen* by
Numba: a Numba-compiled function sees the value of those variables at the
time the function was compiled.  Also, it is not possible to change their
values from the function.

To modify a global variable within a function, you can pass it as an argument
and modify it in place without the need to explicitly return it.

.. todo:: This document needs completing.
