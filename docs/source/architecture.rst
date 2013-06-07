==================
Numba Architecture
==================

.. contents::

Introduction
============

This document serves two purposes: to introduce other developers to
the high-level design of Numba's internals, and as a point for
discussion and synchronization for current Numba developers.

Core Entry Points
=================

Numba has several modes of use:

#. As a run-time translator of Python functions into low-level
   functions.

#. As a call-time specializer of Python functions into low-level
   functions.

#. As a run-time builder of extension types.

#. As a compile-time translator of Python modules into shared object
   libraries.

  #. As a compile-time builder of extension types.

#. As a framework for static analysis and code generation.

The following subsections describe the primary entry points for these
modes of use.  Each usage mode corresponds to a specific set of definitions provided in the top-level numba module.

Run-time Translation
--------------------

.. |jit| replace:: :py:func:`numba.jit`

Users denote run-time translation of a function using the |jit| decorator.

Call-time Specialization
------------------------

.. |autojit| replace:: :py:func:`numba.autojit`

Users denote call-time specialization of a function using the |autojit|
decorator.

Extension Types
---------------

Numba supports building extension types using the |jit| decorator on a class.

Compile-time Translation
------------------------

.. |export| replace:: :py:func:`numba.export`
.. |exportmany| replace:: :py:func:`numba.exportmany`

Users denote compile-time translation of a function using the |export|
and |exportmany| decorators.

Translation Internals
=====================


Towards More Modular Pipelines
------------------------------

The end goal of building a more modular pipeline is to decouple
stages of compilation and make a more modular way of composing
transformations.

- State threaded through the pipeline
    1) AST - Abstract syntax tree, possibly mutated as a
    side-effect of a pass.

    2) Structured Environment - A dict like object which holds
    the intermediate forms and data produced as a result of data.

- Composition of Stages
    - Sequencing
    - Composition Operator
    - Error handling and reporting in pass failure.

- Pre/Post Condition Checking

    - Stages should have attached pre / post conditions to check
      the success criterion of the pass for the inputted or
      resulting ast and environment. Failure to meet this
      conditions should cause the pipeline to halt.

Modularity
~~~~~~~~~~

Note: recursive definitions

::

   jit     := parse o link o jit
   pycc    := parse o emit o link
   autojit := cache o autojit
   cache   := pipeline o jit

   blaze   := mapast o jit

Diagram
~~~~~~~

::

   Block diagram:
                    Input
                       |
   +----------------------+
   |          pass 1      |
   +--------|----------|--+
          context     ast
            |          |
     postcondition     |
            |          |
     precondition      |
            |          |
   +--------|----------|--+
   |          pass 2      |
   +--------|----------|--+
          context     ast
            |          |
     postcondition     |
            |          |
     precondition      |
            |          |
   +--------|----------|--+
   |          pass 3      |
   +--------|----------|--+
          context     ast
            |          |
     precondition      |
            |          |
            +----------+-----> Output


*Discussion: Pipeline Composition*
----------------------------------

.. |Pipeline| replace:: :py:class:`numba.pipeline.Pipeline`

We can do composition in a functional way::

  def compose_stages(stage1, stage2):
    def composition(ast, env):
      return stage2(stage1(ast, env), env)
    return composition

  pipeline = compose_stages(...compose_stages(parse, ...), ...)

Or, we can do composition using iteration::

  for stage in stages:
    ast = stage(ast, env)

Whether the end result is a function or a class is also still up for
discussion.

Proposal 1: We replace the Pipeline class to use a list of stages,
but these can either be functions or subclasses of the
``PipelineStage`` class.


*Discussion: Pipeline Environments*
-----------------------------------

Proposal 1: We present an ad hoc environment.  This provides the most
flexibility for developers to patch the environment as they see fit.

Proposal 2: We present a well defined environment class.  The class
will have well defined properties that are documented and type-checked
when the internal stage checking flag is set.

Terms and Definitions
=====================

Appendix
========


