
CUDA Fast Math
==============

.. _cuda-fast-math:

As noted in :ref:`fast-math` for certain classes of applications that utilize
floating point, strict IEEE-754 conformance is not required. For this subset
of applications, performance speedups may be possible.

The CUDA target implements :ref:`fast-math` behavior with two differences. First
the ``fastmath`` argument is limited to the values ``True`` and ``False``.

Secondly calls to a subset of 32-bit math functions will generate more
efficient code than the non-fastmath equivalent. The list of more efficient
functions is:

* cos
* sin
* tan
* exp
* log2
* log10
* log
* pow