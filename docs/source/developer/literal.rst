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

Numba will type ``a`` as ``Literal[int](123)``. The definition of ``bar()`` can
specialize it's implementation knowing that the second argument is an ``int``
with the value ``123``.

``Literal`` Type
----------------

Classes and methods related to the ``Literal`` type.

.. autoclass:: numba.types.Literal

.. autofunction:: numba.types.literal

.. autofunction:: numba.types.unliteral

.. autofunction:: numba.types.maybe_literal

Hints for Literal Typing
------------------------

To mark the requirement for a ``Literal`` type in jit-code. Use the following
function:

.. autofunction:: numba.literally

Interal Details
~~~~~~~~~~~~~~~

Internally, the compiler raises a ``ForceLiteralArgs`` exception to signal
the dispatcher to wrap specified arguments using the ``Literal`` type.

.. autoclass:: numba.errors.ForceLiteralArg
    :members: __init__, combine, __or__


Inside Extensions
-----------------

``@overload`` extensions can use ``literally`` inside the implementation body
like in normal jit-code.

Explicit handling of literal requirements is possible by using the
followings:

.. autoclass:: numba.extending.SentryLiteralArgs
    :members:

.. autoclass:: numba.extending.BoundLiteralArgs
    :members:

.. autofunction:: numba.extending.sentry_literal_args
