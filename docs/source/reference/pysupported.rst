
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


.. todo:: write this
