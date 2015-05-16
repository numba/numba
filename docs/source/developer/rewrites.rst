=====================================================
Using the Numba Rewrite Pass for Fun and Optimization
=====================================================

Overview
========

This section introduces intermediate representation (IR) rewrites, and
how they can be used to implement optimizations.

As discussed earlier in ":ref:`rewrite-typed-ir`", rewriting the Numba
IR allows us to perform optimizations that would be much more
difficult to perform at the lower LLVM level.  Similar to the Numba
type and lowering subsystems, the rewrite subsystem is user
extensible.  This extensibility affords Numba the possibility of
supporting a wide variety of domain-specific optimizations (DSO's).

The remaining subsections detail the mechanics of implementing a
rewrite, registering a rewrite with the rewrite registry, and provide
examples of adding new rewrites, as well as internals of the array
expression optimization pass.  We conclude by reviewing some use cases
exposed in the examples, as well as reviewing any points where
developers should take care.


Rewriting Passes
================

Rewriting passes have a simple :func:`~Rewrite.match` and
:func:`~Rewrite.apply` interface.  The division between matching and
rewriting follows how one would define a term rewrite in a declarative
domain-specific languages (DSL's).  In such DSL's, one may write a
rewrite as follows::

  <match> => <replacement>


The ``<match>`` and ``<replacement>`` symbols represent IR term
expressions, where the left-hand side presents a pattern to match, and
the right-hand side an IR term constructor to build upon matching.
Whenever the rewrite matches an IR pattern, any free variables in the
left-hand side are bound within a custom environment.  When applied,
the rewrite uses the pattern matching environment to bind any free
variables in the right-hand side.

As Python is not commonly used in a declarative capacity, Numba uses
object state to handle the transfer of information between the
matching and application steps.


The :class:`Rewrite` Base Class
-------------------------------

.. class:: Rewrite

   The :class:`Rewrite` class simply defines an abstract base class
   for Numba rewrites.  Developers should define rewrites as
   subclasses of this base type, overloading the
   :func:`~Rewrite.match` and :func:`~Rewrite.apply` methods.

   .. attribute:: pipeline

       The pipeline attribute contains the
       :class:`numba.compiler.Pipeline` instance that is currently
       compiling the function under consideration for rewriting.

   .. method:: __init__(self, pipeline, *args, **kws)

       The base constructor for rewrites simply stashes its arguments
       into attributes of the same name.  Unless being used in
       debugging or testing, rewrites should only be constructed by
       the :class:`RewriteRegistry` in the
       :func:`RewriteRegistry.apply` method, and the construction
       interface should remain stable (though the pipeline will
       commonly contain just about everything there is to know).

   .. method:: match(self, block, typemap, callmap)

      The :func:`~Rewrite.match` method takes three arguments other
      than *self*:

      * *block*: This is an instance of :class:`numba.ir.Block`.  The
        matching method should iterate over the instructions contained
        in the :attr:`numba.ir.Block.body` member.

      * *typemap*: This is a Python :class:`dict` instance mapping
        from symbol names in the IR, represented as strings, to Numba
        types.

      * *callmap*: This is another :class:`dict` instance mapping from
        calls, represented as :class:`numba.ir.Expr` instances, to
        their corresponding call site type signatures, represented as
        a :class:`numba.typing.templates.Signature` instance.

      The :func:`~Rewrite.match` method should return a :class:`bool`
      result.  A ``True`` result should indicate that one or more
      matches were found, and the :func:`~Rewrite.apply` method will
      return a new replacement :class:`numba.ir.Block` instance.  A
      ``False`` result should indicate that no matches were found, and
      subsequent calls to :func:`~Rewrite.apply` will return undefined
      or invalid results.

   .. method:: apply(self)

      The :func:`~Rewrite.apply` method should only be invoked
      following a successful call to :func:`~Rewrite.match`.  This
      method takes no additional parameters other than *self*, and
      should return a replacement :class:`numba.ir.Block` instance.

      As mentioned above, the behavior of calling
      :func:`~Rewrite.apply` is undefined unless
      :func:`~Rewrite.match` has already been called and returned
      ``True``.


Subclassing :class:`Rewrite`
----------------------------

Before going into the expectations for the overloaded methods any
:class:`Rewrite` subclass must have, let's step back a minute to
review what is taking place here.  By providing an extensible
compiler, Numba opens itself to user-defined code generators which may
be incomplete, or worse, incorrect.  When a code generator goes awry,
it can cause abnormal program behavior or early termination.
User-defined rewrites add a new level of complexity because they must
not only generate correct code, but the code they generate should
ensure that the compiler does not get stuck in a match/apply loop.
Non-termination by the compiler will directly lead to non-termination
of user function calls.

There are several ways to help ensure that a rewrite terminates:

* *Typing*: A rewrite should generally attempt to decompose composite
  types, and avoid composing new types.  If the rewrite is matching a
  specific type, changing expression types to a lower-level type will
  ensure they will no long match after the rewrite is applied.

* *Special instructions*: A rewrite may synthesize custom operators or
  use special functions in the target IR.  This technique again
  generates code that is no longer within the domain of the original
  match, and the rewrite will terminate.

In the ":ref:`case-study-array-expressions`" subsection, below, we'll
see how the array expression rewriter uses both of these techiniques.


Overloading :func:`Rewrite.match`
---------------------------------

Every rewrite developer should seek to have their implementation of
:func:`~Rewrite.match` return a ``False`` value as quickly as
possible.  Numba is a just-in-time compiler, and adding compilation
time ultimately adds to the user's run time.  When a rewrite returns
``False`` for a given block, the registry will no longer process that
block with that rewrite, and the compiler is that much closer to
proceeding to lowering.

This need for timeliness has to be balanced against collecting the
necessary information to make a match for a rewrite.  Rewrite
developers should be comfortable adding dynamic attributes to their
subclasses, and then having these new attributes guide construction of
the replacement basic block.


Overloading :func:`Rewrite.apply`
-----------------------------------

The :func:`~Rewrite.apply` method should return a replacement
:class:`numba.ir.Block` instance to replace the basic block that
contained a match for the rewrite.  As mentioned above, the IR built
by :func:`~Rewrite.apply` methods should preserve the semantics of the
user's code, but also seek to avoid generating another match for the
same rewrite or set of rewrites.


The Rewrite Registry
====================

When you want to include a rewrite in the rewrite pass, you should
register it with the rewrite registry.  The :mod:`numba.rewrites`
module provides both the abstract base class and a class decorator for
hooking into the Numba rewrite subsystem.  The following illustrates a
stub definition of a new rewrite::

  from numba import rewrites

  @rewrites.register_rewrite
  class MyRewrite(rewrites.Rewrite):

      def match(self, block, typemap, calltypes):
          raise NotImplementedError("FIXME")

      def apply(self):
          raise NotImplementedError("FIXME")


Developers should note that using the class decorator as shown above
will register a rewrite at import time.  It is the developer's
responsibility to ensure their extensions are loaded before
compilation starts.


Case study: Constant Folding
============================

This subsection discusses how constant folding might be added in
support of a user-defined type.


.. _`case-study-array-expressions`:

Case study: Array Expressions
=============================

This subsection looks at the array expression rewriter in more depth.


Conclusions and Caveats
=======================

This section reviews rewrites, and provides guidance for possible
stumbling blocks when using them.
