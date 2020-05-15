.. _developer-literally:

======================
Notes on Literal Types
======================

.. note:: This document describes an advanced feature designed to overcome
          some limitations of the compilation mechanism relating to types.

Some features need to specialize based on the literal value during
compliation to produce type stable code necessary for successful compilation in
Numba. This can be achieved by propagating the literal value through the type
system. Numba recognizes inline literal values as :class:`numba.types.Literal`.
For example::

    def foo(x):
        a = 123
        return bar(x, a)

Numba will infer the type of ``a`` as ``Literal[int](123)``. The definition of
``bar()`` can subsequently specialize its implementation knowing that the
second argument is an ``int`` with the value ``123``.

``Literal`` Type
----------------

Classes and methods related to the ``Literal`` type.

.. autoclass:: numba.types.Literal

.. autofunction:: numba.types.literal

.. autofunction:: numba.types.unliteral

.. autofunction:: numba.types.maybe_literal

Specifying for Literal Typing
-----------------------------

To specify a value as a ``Literal`` type in code scheduled for JIT compilation,
use the following function:

.. autofunction:: numba.literally

Code Example
~~~~~~~~~~~~

.. literalinclude:: ../../../numba/tests/doc_examples/test_literally_usage.py
   :language: python
   :caption: from ``test_literally_usage`` of ``numba/tests/doc_examples/test_literally_usage.py``
   :start-after: magictoken.ex_literally_usage.begin
   :end-before: magictoken.ex_literally_usage.end
   :dedent: 4
   :linenos:


Internal Details
~~~~~~~~~~~~~~~~

Internally, the compiler raises a ``ForceLiteralArgs`` exception to signal
the dispatcher to wrap specified arguments using the ``Literal`` type.

.. autoclass:: numba.errors.ForceLiteralArg
    :members: __init__, combine, __or__


Inside Extensions
-----------------

``@overload`` extensions can use ``literally`` inside the implementation body
like in normal jit-code.

Explicit handling of literal requirements is possible through use of the
following:

.. autoclass:: numba.extending.SentryLiteralArgs
    :members:

.. autoclass:: numba.extending.BoundLiteralArgs
    :members:

.. autofunction:: numba.extending.sentry_literal_args
