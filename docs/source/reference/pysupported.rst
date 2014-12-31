
=========================
Supported Python features
=========================

Language
========

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

Standard library modules
========================


.. todo:: write this
