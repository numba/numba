=========================================================
Using the No-Python Rewrite Pass for Fun and Optimization
=========================================================

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
:func:`~Rewrite.apply` interface.  This division comes from
declarative domain-specific languages (DSL's) for defining term
rewrites.  In such DSL's, one may write a rewrite as follows::

  <match> => <replacement>


The ``<match>`` and ``<replacement>`` symbols represent IR term
expressions, where the left-hand side presents a pattern to match, and
the right-hand side an IR term constructor.  Whenever the rewrite
matches an IR pattern, any free variables in the left-hand side are
bound within a custom environment.  When applied, the rewrite uses the
pattern matching environment to bind any free variables in the
right-hand side.

As Python is not commonly used in a declarative capacity, Numba uses
object state to handle the hand-off of information on a particular
match.


The :class:`Rewrite` Base Class
-------------------------------

.. class:: Rewrite

   .. method:: match(self, block, typemap, callmap)

   .. method:: apply(self)


Overloading :func:`Rewrite.match`
---------------------------------


Overloading :func:`Rewrite.apply`
-----------------------------------


The Rewrite Registry
====================

When you want to include a rewrite in the rewrite pass, you should
register it with the rewrite registry.


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
