
Deviations from Python semantics
================================


Integer width
-------------

While Python has arbitrary-sized integers, integers in Numba-compiled
functions get a fixed size through :term:`type inference` (usually,
the size of a machine integer).  This means that arithmetic
operations can wrapround or produce undefined results or overflow.

Type inference can be overriden by an explicit type specification,
if fine-grained control of integer width is desired.

.. seealso::
   :ref:`Enhancement proposal 1: Changes in integer typing <nbep-1>`


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

Numba **may or may not** copy global variables referenced inside a compiled
function.  Small global arrays are copied for potential compiler optimization
with immutability assumption.  However, large global arrays are not copied to
conserve memory.  The definition of "small" and "large" may change.


.. todo:: This document needs completing.
