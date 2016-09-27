========================
NBEP ?: Typing Recursion
========================

:Author: Siu Kwan Lam
:Date: Sept 2016
:Status: Draft

Introduction
============

This document proposes an enhancement to the type inference to
support recursion without explicit annotating the function signature.
As a result, the proposal enables numba to type-infer both self-recursive and
mutual-recursive functions under some limitations.  In practice, these
limitions can be easily overcome by specifying a compilation order.


The Current State
=================

Recursion support in numba is currently limited to self-recursion with explicit
type annotation for the function.  This limitation comes from the inability to
determine the return type of a recursive call.  This is because the callee is
either the current function (for self-recursion) or a parent function
(mutual-recursion).  Its type inference has been suspended while waiting for
the function-type of its callee.  This forms a cyclic dependency.
Given a function ``foo()`` that calls ``bar()``, which in turns call ``foo()``::

    def foo(x):
        if x > 0:
            return bar(x)
        else:
            return 1

    def bar(x):
        return foo(x - 1)


The type inference of ``foo()`` depends on that of ``bar()``, which depends on
``foo()``.  Therefore ``foo()`` depends on itself and the type inference cannot
terminate.


The Solution
============

The proposed solution has two components:

1. Introduce a compile-time *callstack* that tracks the compiling functions.
2. Allow partial type inference on functions by leveraging the return type
   on non-recursive control-flow paths.

The compile-time callstack stores typing information of functions being
compiled.  Like an ordinary callstack, it pushes a new record every time a
function is "called".  Since this occurs at compile-time, a "call" triggers
a compilation of the callee.

To detect recursion, the compile-time callstack is searched bottom-up
(stack grows downward) for a record that matches the callee.
The record contains a reference to the type inference state.
With that, type inference can be resumed to determine the return type.

Recall that the type inference cannot be resumed normally because of the cyclic
dependency of the return type.  In practice, we can assume that a useful
program must have a terminating condition, a path that does not recurse.  So,
type inference can make an initial guess for the return-type at the recursive
call by using the return-type determined by the non-recursive paths.  This
allows type information to propagate on the recursive paths to generate the
final return type, which is used to refine the type information by the
subsequent iteration in the type inference algorithm.


The following figure illustrates the compile-time callstack when the compiler
reaches the recursive call to ``foo()`` from ``bar()``:

.. image:: recursion_callstack.svg
    :width: 400px

At this time, the type inference of ``foo()`` is suspended and that of ``bar()``
is active.  The compiler can see that the callee is already compiling by
searching the callstack.  Knowing that it is a recursive call, the compiler
can resume the type-inference on ``foo()`` by ignoring the paths that contain
recursive calls.  This means only the ``else`` branch is considered and we can
easily tell that ``foo()`` returns an ``int`` in this case.  The compiler will
then set the initial return type of ``foo()`` and ``bar()`` to ``int``.  The
subsequent type propagation can use this information to complete the type
inference of both functions, unifying the return-type of all returning paths.


Limitations
===========

For the proposed type inference algorithm to terminate, it assumes that
at least one of the control path leads to a return-statement without doing
a recursive call.  Otherwise, the algorithm will raise an exception indicating
a potential runaway recursion.

For example::

    @jit
    def first(x):
        # The recursing call must have a path that is non-recursing.
        if x > 0:
            return second(x)
        else:
            return 1

    @jit
    def second(x):
        return third(x)

    @jit
    def third(x):
        return first(x - 1)


The ``first()`` function must be the compiled first for the type inference to
complete successfully.  Compiling any other function first will lead to failure
in type inference.  The type inferencer will treat it as a runaway recursion
due to the lack of a non-recursive exit in the recursive callee.

For example, compiling ``second()`` first will move the recursive call to
``first()``.  When the compiler tries to resume the type inference of
``second()``, it will fail to find a non-recursive path.

This is a small limitation and can be overcome easily by code restructuring or
precompiling in a specific order.

