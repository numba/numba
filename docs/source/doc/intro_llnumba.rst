===================
Introducing LLNumba
===================

In this article, we introduce the llnumba sub-package.  The primary
goal of the llnumba sub-package is to provide a Python dialect/subset
that maps directly to LLVM code.  LLNumba differs from Numba in the
following aspects:

  * LLNumba code is not intended to work in Python if not translated
    and wrapped.
  * The LLNumba translator only uses LLVM types.
  * LLNumba is explicitly typed, and does not support type inference.
    LLNumba does not support implicit casts, all casts must be explicit.
  * LLNumba supports code that directly calls the C API, the Python C
    API, and the llvm.core.Builder methods.

Additionally, we designed the sub-package to have the following
engineering properties:

  * Usable from Python 2.7, and 3.X.  At the time of writing, we plan
    to support Python 2.6.
  * Clean from numba dependencies (other than llvmpy), and can be used
    as a standalone code generator without a full Numba installation.
  * Provides a series of Python bytecode passes that can be easily
    used by other projects.


LLNumba Origins
===============

We developed LLNumba with the initial goal of simplifying writing
LLVM-specific code in Numba.


LLNumba Internals
=================

In this section, we describe the various passes performed by the
LLNumba translator.


Conclusions
===========

LLNumba is neat.
