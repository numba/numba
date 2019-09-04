.. _developer-literally:

======================
Notes on Literal Types
======================

.. note:: This document describes an advanced feature designed to overcome
          limitation of the compiler system.

Some features need to specialize based on the literal value during
compliation to produce type stable code necessary for successful compilation in
Numba. This can be achieved by propagating the literal value through the type
system. Numba recognizes inline literal value as :class:`numba.types.Literal`.
For example::

    def foo(x):
        a = 123
        return bar(x, a)

Numba will type `a` as ``Literal[int](123)``. The definition of ``bar()`` can
specialize it's implementation knowing that the second argument is a ``123``.


.. autofunction:: numba.literally


.. autoclass:: numba.types.Literal


.. autoclass:: numba.errors.ForceLiteralArg

