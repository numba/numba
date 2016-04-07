========================================
Supported Python features in CUDA Python
========================================

This page lists the Python features supported in the CUDA Python.  This includes
all kernel and device functions compiled with ``@cuda.jit`` and other higher
level Numba decorators that targets the CUDA GPU.

Language
========

Execution Model
---------------

CUDA Python maps directly to the *single-instruction multiple-thread*
execution (SIMT) model of CUDA.  Each instruction is implicitly
executed by multiple threads in parallel.  With this execution model, array
expressions are less useful because we don't want multiple threads to perform
the same task.  Instead, we want threads to perform a task in a cooperative
fashion.

For details please consult the
`CUDA Programming Guide
<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model>`_.

Constructs
----------

The following Python constructs are not supported:

* Exception handling (``try .. except``, ``try .. finally``)
* Context management (the ``with`` statement)
* Comprehensions (either list, dict, set or generator comprehensions)
* Generator (any ``yield`` statements)

The ``raise`` and ``assert`` statements are supported.
See :ref:`nopython language support <pysupported-language>`.

Built-in types
===============

The following built-in types support are inherited from CPU nopython mode.

* int
* float
* complex
* bool
* None
* tuple

See :ref:`nopython built-in types <pysupported-builtin-types>`.


Built-in functions
==================

The following built-in functions are supported:

* :func:`abs`
* :class:`bool`
* :class:`complex`
* :func:`enumerate`
* :class:`float`
* :class:`int`: only the one-argument form
* :func:`len`
* :func:`min`: only the multiple-argument form
* :func:`max`: only the multiple-argument form
* :class:`range`: semantics are similar to those of Python 3 even in Python 2:
  a range object is returned instead of an array of values.
* :func:`round`
* :func:`zip`


Standard library modules
========================


``cmath``
---------

The following functions from the :mod:`cmath` module are supported:

* :func:`cmath.acos`
* :func:`cmath.acosh`
* :func:`cmath.asin`
* :func:`cmath.asinh`
* :func:`cmath.atan`
* :func:`cmath.atanh`
* :func:`cmath.cos`
* :func:`cmath.cosh`
* :func:`cmath.exp`
* :func:`cmath.isfinite`
* :func:`cmath.isinf`
* :func:`cmath.isnan`
* :func:`cmath.log`
* :func:`cmath.log10`
* :func:`cmath.phase`
* :func:`cmath.polar`
* :func:`cmath.rect`
* :func:`cmath.sin`
* :func:`cmath.sinh`
* :func:`cmath.sqrt`
* :func:`cmath.tan`
* :func:`cmath.tanh`

``math``
--------

The following functions from the :mod:`math` module are supported:

* :func:`math.acos`
* :func:`math.asin`
* :func:`math.atan`
* :func:`math.arctan`
* :func:`math.acosh`
* :func:`math.asinh`
* :func:`math.atanh`
* :func:`math.cos`
* :func:`math.sin`
* :func:`math.tan`
* :func:`math.hypot`
* :func:`math.cosh`
* :func:`math.sinh`
* :func:`math.tanh`
* :func:`math.atan2`
* :func:`math.erf`
* :func:`math.erfc`
* :func:`math.exp`
* :func:`math.expm1`
* :func:`math.fabs`
* :func:`math.gamma`
* :func:`math.lgamma`
* :func:`math.log`
* :func:`math.log10`
* :func:`math.log1p`
* :func:`math.sqrt`
* :func:`math.pow`
* :func:`math.ceil`
* :func:`math.floor`
* :func:`math.copysign`
* :func:`math.fmod`
* :func:`math.isnan`
* :func:`math.isinf`


``operator``
------------

The following functions from the :mod:`operator` module are supported:

* :func:`operator.add`
* :func:`operator.and_`
* :func:`operator.div` (Python 2 only)
* :func:`operator.eq`
* :func:`operator.floordiv`
* :func:`operator.ge`
* :func:`operator.gt`
* :func:`operator.iadd`
* :func:`operator.iand`
* :func:`operator.idiv` (Python 2 only)
* :func:`operator.ifloordiv`
* :func:`operator.ilshift`
* :func:`operator.imod`
* :func:`operator.imul`
* :func:`operator.invert`
* :func:`operator.ior`
* :func:`operator.ipow`
* :func:`operator.irshift`
* :func:`operator.isub`
* :func:`operator.itruediv`
* :func:`operator.ixor`
* :func:`operator.le`
* :func:`operator.lshift`
* :func:`operator.lt`
* :func:`operator.mod`
* :func:`operator.mul`
* :func:`operator.ne`
* :func:`operator.neg`
* :func:`operator.not_`
* :func:`operator.or_`
* :func:`operator.pos`
* :func:`operator.pow`
* :func:`operator.rshift`
* :func:`operator.sub`
* :func:`operator.truediv`
* :func:`operator.xor`
