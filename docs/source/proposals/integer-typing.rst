=================================
NBEP 1: Changes in integer typing
=================================

:Author: Antoine Pitrou
:Date: July 2015
:Status: Final


Current semantics
=================

Type inference of integers in Numba currently has some subtleties
and some corner cases.  The simple case is when some variable has an obvious
Numba type (for example because it is the result of a constructor call to a
Numpy scalar type such as ``np.int64``). That case suffers no ambiguity.

The less simple case is when a variable doesn't bear such explicit
information.  This can happen because it is inferred from a built-in Python
``int`` value, or from an arithmetic operation between two integers, or
other cases yet.  Then Numba has a number of rules to infer the resulting
Numba type, especially its signedness and bitwidth.

Currently, the generic case could be summarized as: *start small,
grow bigger as required*.  Concretely:

1. Each constant or pseudo-constant is inferred using the *smallest signed
   integer type* that can correctly represent it (or, possibly, ``uint64``
   for positive integers between ``2**63`` and ``2**64 - 1``).
2. The result of an operation is typed so as to ensure safe representation
   in the face of overflow and other magnitude increases (for example,
   ``int32 + int32`` would be typed ``int64``).
3. As an exception, a Python ``int`` used as function argument is always
   typed ``intp``, a pointer-size integer.  This is to avoid the proliferation
   of compiled specializations, as otherwise various integer bitwidths
   in input arguments may produce multiple signatures.

.. note::
   The second rule above (the "respect magnitude increases" rule)
   reproduces Numpy's behaviour with arithmetic on scalar values.
   Numba, however, has different implementation and performance constraints
   than Numpy scalars.

   It is worth nothing, by the way, that Numpy arrays do not implement
   said rule (i.e. ``array(int32) + array(int32)`` is typed ``array(int32)``,
   not ``array(int64)``).  Probably because this makes performance more
   controllable.

This has several non-obvious side-effects:

1. It is difficult to predict the precise type of a value inside a function,
   after several operations.  The basic operands in an expression tree
   may for example be ``int8`` but the end result may be ``int64``.  Whether
   this is desirable or not is an open question; it is good for correctness,
   but potentially bad for performance.

2. In trying to follow the correctness over predictability rule, some values
   can actually leave the integer realm.  For example, ``int64 + uint64``
   is typed ``float64`` in order to avoid magnitude losses (but incidentally
   will lose precision on large integer values...), again following Numpy's
   semantics for scalars.  This is usually not intended by the user.

3. More complicated scenarios can produce unexpected errors at the type unification
   stage.  An example is at `Github issue 1299 <https://github.com/numba/numba/issues/1299>`_,
   the gist of which is reproduced here::

      @jit(nopython=True)
      def f():
          variable = 0
          for i in range(1):
              variable = variable + 1
          return np.arange(variable)

   At the time of this writing, this fails compiling, on a 64-bit system,
   with the error::

      numba.errors.TypingError: Failed at nopython (nopython frontend)
      Can't unify types of variable '$48.4': $48.4 := {array(int32, 1d, C), array(int64, 1d, C)}

   People expert with Numba's type unification system can understand why.
   But the user is caught in mystery.


Proposal: predictable width-conserving typing
=============================================

We propose to turn the current typing philosophy on its head.  Instead
of "*start small and grow as required*", we propose "*start big and keep
the width unchanged*".

Concretely:

1. The typing of Python ``int`` values used as function arguments doesn't
   change, as it works satisfyingly and doesn't surprise the user.

2. The typing of integer *constants* (and pseudo-constants) changes to match
   the typing of integer arguments.  That is, every non-explicitly typed
   integer constant is typed ``intp``, the pointer-sized integer; except for
   the rare cases where ``int64`` (on 32-bit systems) or ``uint64`` is
   required.

3. Operations on integers promote bitwidth to ``intp``, if smaller, otherwise
   they don't promote.  For example, on a 32-bit machine, ``int8 + int8``
   is typed ``int32``, as is ``int32 + int32``.  However, ``int64 + int64``
   is typed ``int64``.

4. Furthermore, mixed operations between signed and unsigned fall back to
   signed, while following the same bitwidth rule.  For example, on a
   32-bit machine, ``int8 + uint16`` is typed ``int32``, as is
   ``uint32 + int32``.


Proposal impact
===============

Semantics
---------

With this proposal, the semantics become clearer.  Regardless of whether
the arguments and constants of a function were explicitly typed or not,
the results of various expressions at any point in the function have
easily predictable types.

When using built-in Python ``int``, the user gets acceptable magnitude
(32 or 64 bits depending on the system's bitness), and the type remains
the same accross all computations.

When explicitly using smaller bitwidths, intermediate results don't
suffer from magnitude loss, since their bitwidth is promoted to ``intp``.

There is also less potential for annoyances with the type unification
system as demonstrated above.  The user would have to force several
different types to be faced with such an error.

One potential cause for concern is the discrepancy with Numpy's scalar
semantics; but at the same time this brings Numba scalar semantics closer
to array semantics (both Numba's and Numpy's), which seems a desirable
outcome as well.

It is worth pointing out that some sources of integer numbers, such
as the ``range()`` built-in, always yield 32-bit integers or larger.
This proposal could be an opportunity to standardize them on ``intp``.

Performance
-----------

Except in trivial cases, it seems unlikely that the current "best fit"
behaviour for integer constants really brings a performance benefit.  After
all, most integers in Numba code would either be stored in arrays (with
well-known types, chosen by the user) or be used as indices, where a ``int8``
is highly unlikely to fare better than a ``intp`` (actually, it may be worse,
if LLVM isn't able to optimize away the required sign-extension).

As a side note, the default use of ``intp`` rather than ``int64``
ensures that 32-bit systems won't suffer from poor arithmetic performance.

Implementation
--------------

Optimistically, this proposal may simplify some Numba internals a bit.
Or, at least, it doesn't threaten to make them significantly more complicated.

Limitations
-----------

This proposal doesn't really solve the combination of signed and unsigned
integers.  It is geared mostly at solving the bitwidth issues, which are
a somewhat common cause of pain for users.  Unsigned integers are in
practice very uncommon in Numba-compiled code, except when explicitly
asked for, and therefore much less of a pain point.

On the bitwidth front, 32-bit systems could still show discrepancies based
on the values of constants: if a constant is too large to fit in 32 bits,
it is typed ``int64``, which propagates through other computations.
This would be a reminiscence of the current behaviour, but rarer and much
more controlled still.

Long-term horizon
-----------------

While we believe this proposal makes Numba's behaviour more regular and more
predictable, it also pulls it further from general compatibility with pure
Python semantics, where users can assume arbitrary-precision integers without
any truncation issues.
