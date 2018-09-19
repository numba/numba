============================================================
Callback into the Python Interpreter from within JIT'ed code
============================================================

There are rare but real cases when a nopython-mode function needs to callback
into the Python interpreter to invoke code that cannot be compiled by Numba.
Such cases include:

- logging progress for long running JIT'ed functions;
- use data structures that are not currently supported by Numba;
- debugging inside JIT'ed code using the Python debugger.

When Numba callbacks into the Python interpreter, the following has to happen:

- acquire the GIL;
- convert values in native representation back into Python objects;
- call-back into the Python interpreter;
- convert returned values from the Python-code into native representation;
- release the GIL.

These steps can be expensive.  Users **should not** rely on the feature
described here on performance-critical paths.


.. _with_objmode:

The ``objmode`` context-manager
===============================

.. warning:: This feature can be easily mis-used.  Users should first consider
    alternative approaches to achieve their intended goal before using
    this feature.

.. autofunction:: numba.objmode
