.. _generated-jit:

================================================
Flexible specializations with ``@generated_jit``
================================================


While the :func:`~numba.jit` decorator is useful for many situations,
sometimes you want to write a function that has different implementations
depending on its input types.  The :func:`~numba.generated_jit` decorator
allows the user to control the selection of a specialization at compile-time,
while fulling retaining runtime execution speed of a JIT function.


Example
=======

Suppose you want to write a function which returns whether a given value
is a "missing" value according to certain conventions.  For the sake of
the example, let's adopt the following definition:

- for floating-point arguments, a missing value is a ``NaN``
- for Numpy datetime64 and timedelta64 arguments, a missing value is a ``NaT``
- other types don't have the concept of a missing value.

That compile-time logic is easily implemented using the
:func:`~numba.generated_jit` decorator::

   import numpy as np

   from numba import generated_jit, types

   @generated_jit(nopython=True)
   def is_missing(value):
       """
       Return True if the value is missing, False otherwise.
       """
       if isinstance(value, types.Float):
           return lambda x: np.isnan(x)
       elif isinstance(value, (types.NPDatetime, types.NPTimedelta)):
           # The corresponding Not-a-Time value
           missing = value('NaT')
           return lambda x: x == missing
       else:
           return lambda x: False


There are several things to note here:

* The decorated function is called with the :ref:`Numba types <numba-types>`
  of the arguments, not their values.

* The decorated function doesn't actually compute a result, it returns
  a callable implementing the actual definition of the function for the
  given types.

* It is possible to pre-compute some data at compile-time (the ``missing``
  variable above) to have them reused inside the compiled implementation.


Compilation options
===================

The :func:`~numba.generated_jit` decorator supports the same keyword-only
arguments as the :func:`~numba.jit` decorator, for example the ``nopython``
and ``cache`` options.

