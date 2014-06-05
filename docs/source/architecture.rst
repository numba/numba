==================
Numba Architecture
==================

.. contents::

Introduction
============

This document serves two purposes: to introduce other developers to
the high-level design of Numba's internals, and as a point for
discussion and synchronization for current Numba developers.


Compile entry point
===================

The module ``numba.compiler`` contains the compiler entry-point
``compile_extra``.
All decorators eventually call this function.  It can be broken down into the
following phrases:

* 1. analyze bytecode
    * gather control-flow and data-flow information
* 2. translate into an intermediate representation
* 3. infer types using local type inference
* 4a. if (3) succeed, lower to native code (native mode)
* 4b. otherwise, assume object type for all variables in the lowering
  (object mode)
    * may try to extract loops in the function and recursively compile the
      extracted loop. (loop-jitting)

Type inference (3) will only succeed if all types have a non-object
type.

Lowering
========

The module ``numba.lowering`` defines the lowering of the internal IR
into LLVM.  The lowering logic is independent of the target.  Each target
provides a target context that describes the target-specific lowering.

Targets
=======

Each target implementation has two contexts:

* a typing context
* a codegen context

The typing context is equivalent to a C header file.
The codegen context is equivalent to a C source file.
This allows the same typing rules to be shared across multiple targets,
while each of them has a different implementation.

The default type context is defined in the ``numba.typing`` subpackage.
The default codegen context is defined in the ``numba.targets`` subpackage.
