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


Zero initialization of variables
--------------------------------

Numba does not track variable liveness at runtime. For simplicity of
implementation, all variables are zero-initialized. Example::

    from numba import njit

    @njit
    def foo():
        for i in range(0):
            pass
        print(i) # will print 0 and not raise UnboundLocalError

    foo()


Object identity
---------------

In :term:`nopython mode`, values are converted ("boxed") from their native
representation back into Python objects whenever they cross the boundary
from compiled code to the interpreter, such as when a jitted function
returns. Numba does not generally preserve Python object identity across
this boundary - if the same value is returned more than once, each
occurrence may be boxed into a *different* Python object::

    from numba import njit

    @njit
    def f():
        a = [1, 2, 3]
        return a, a

    x, y = f()
    x is y  # False

This applies to values created inside a jitted function, regardless of
their type - arrays, :ref:`@jitclass <jitclass>` instances, lists, sets,
and scalars are all affected.

.. warning::
   For arrays and jitclass instances, two boxed objects that are not
   identical (``x is y`` is ``False``) can nonetheless refer to the *same*
   underlying memory, because the underlying buffer is not copied when
   boxing occurs more than once for the same value::

       import numpy as np

       @njit
       def f_returning_array_twice():
           arr = np.array([1, 2, 3])
           return arr, arr

       x, y = f_returning_array_twice()
       x is y      # False
       x[0] = 99
       y[0]        # 99 - the underlying buffer is shared

   This means ``is`` cannot be used to reliably check whether two arrays
   or jitclass instances returned from a jitted function alias the same
   memory; use :func:`numpy.shares_memory` for arrays instead.

   Lists and sets do not share this behavior - independently boxed copies
   of a list or set created inside a jitted function do not share
   underlying storage, so mutating one does not affect the other.

A value that is *passed into* a jitted function as an argument is not
re-boxed on return - if it is returned unchanged, the exact same Python
object that was passed in is returned::

    @njit
    def identity(x):
        return x

    original = [1, 2, 3]
    identity(original) is original  # True

Numba's implementation of the ``is`` operator also differs from CPython's
for immutable types (such as integers, floats, and tuples of immutable
values): it compares by value rather than by identity, which can produce
results that CPython does not guarantee for non-cached objects::

    @njit
    def same(u, v):
        return u is v

    same(666, 666)  # True in Numba; not guaranteed by CPython
