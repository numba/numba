
.. _low-level-extending:

Low-level extension API
=======================

This extension API is available through the :mod:`numba.extending` module.
It allows you to hook directly into the Numba compilation chain.  As such,
it distinguished between several compilation phases:

* The *typing* phase deduces the types of variables in a compiled function
  by looking at the operations performed.

* The *lowering* phase converts high-level Python operations into low-level
  LLVM code.  This phase feeds on the typing information derived by the
  typing phase.

* *Boxing* and *unboxing* convert Python objects into native values, and
  vice-versa.  They occur at the boundaries of calling a Numba function
  from the Python interpreter.


Typing
------

Broadly speaking, typing comes in two flavours: typing plain *values*
(e.g. Python global variables) and typing *operations* (or *functions*)
on known value types.

However, before typing anything, you may need to declare your own type first.
Numba doesn't recognize Python types automatically.  While the Numba type
hierarchy often parallels Python types, it is entirely separate, for
flexibility and expressivity reasons.  This means you have to define a Numba type
(or, quite often, a type class) for any new Python type you want to teach
Numba about.

