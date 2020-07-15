.. _pysemantics:

Deviations from Python Semantics
================================

Bounds Checking
---------------

By default, instead of causing an :class:`IndexError`, accessing an
out-of-bound index of an array in a Numba-compiled function will return
invalid values or lead to an access violation error (it's reading from
invalid memory locations). Bounds checking can be enabled on a specific
function via the :ref:`boundscheck <jit-decorator-boundscheck>`
option of the jit decorator. Additionally, the :envvar:`NUMBA_BOUNDSCHECK`
can be set to 0 or 1 to globally override this flag.

.. note::
  Bounds checking will slow down typical functions so it is recommended to only
  use this flag for debugging purposes.

Exceptions and Memory Allocation
--------------------------------

Due to limitations in the current compiler when handling exceptions, memory
allocated (almost always NumPy arrays) within a function that raises an
exception will **leak**.  This is a known issue that will be fixed, but in the
meantime, it is best to do memory allocation outside of functions that can
also raise exceptions.

Integer width
-------------

While Python has arbitrary-sized integers, integers in Numba-compiled
functions get a fixed size through :term:`type inference` (usually,
the size of a machine integer).  This means that arithmetic
operations can wrapround or produce undefined results or overflow.

Type inference can be overridden by an explicit type specification,
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
