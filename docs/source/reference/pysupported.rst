
=========================
Supported Python features
=========================

Language
========

Constructs
----------

Numba strives to support as much of the Python language as possible, but
some language features are not available inside Numba-compiled functions:

* Function definition
* Class definition
* Exception handling (``try .. except``, ``try .. finally``)
* Context management (the ``with`` statement)

The ``raise`` statement is only supported in the simplest form of
raising a type without explicitly creating an instance, i.e.
``raise TypeError`` is possible but not ``raise TypeError("some message")``.

Similarly, the ``assert`` statement is only supported without an explicit
error message.

Function calls
--------------

Numba only supports function calls using positional arguments.
``*args`` and ``**kwargs`` are not supported.


Built-in types
==============

int, bool
---------

Arithmetic operations as well as truth values are supported.

The following attributes and methods are supported:

* ``.conjugate()``
* ``.real``
* ``.imag``

float, complex
--------------

Arithmetic operations as well as truth values are supported.

The following attributes and methods are supported:

* ``.conjugate()``
* ``.real``
* ``.imag``

tuple
-----

Tuple construction and unpacking is supported.

None
----

The :const:`None` value is supported for identity testing (when using
an :class:`~numba.optional` type).


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
* :func:`print`: only numbers and strings; no ``file`` or ``sep`` argument
* :class:`range`
* :func:`round`: only the two-argument form
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

``ctypes``
----------

``math``
--------

The following functions from the :mod:`math` module are supported:

* :func:`math.acos`
* :func:`math.acosh`
* :func:`math.asin`
* :func:`math.asinh`
* :func:`math.atan`
* :func:`math.atan2`
* :func:`math.atanh`
* :func:`math.ceil`
* :func:`math.copysign`
* :func:`math.cos`
* :func:`math.cosh`
* :func:`math.degrees`
* :func:`math.exp`
* :func:`math.expm1`
* :func:`math.fabs`
* :func:`math.floor`
* :func:`math.hypot`
* :func:`math.isfinite`
* :func:`math.isinf`
* :func:`math.isnan`
* :func:`math.log`
* :func:`math.log10`
* :func:`math.log1p`
* :func:`math.pow`
* :func:`math.radians`
* :func:`math.sin`
* :func:`math.sinh`
* :func:`math.sqrt`
* :func:`math.tan`
* :func:`math.tanh`
* :func:`math.trunc`


.. todo:: finish this
