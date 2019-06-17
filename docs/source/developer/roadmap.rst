=====================
Numba Project Roadmap
=====================

.. note::
    This page was last revised in *December 2018*.

This roadmap is for informational purposes only.  Priorities and resources
change, so we may choose to reorder or abandon things on this list.
Additionally, the further out items are, the less concrete they will be.  If
you have an interest in working on one of these items, please open an issue
where we can discuss the design and approach first.

Short Term: 2019H1
==================

* Container improvements:

  * Numba dictionary support
  * Refactor lists to follow new container best practices.
    See the discussion in `issue 3546 <https://github.com/numba/numba/issues/3546#issuecomment-443008201>`_.

* Deprecate Python 2.7 support
* Improve caching:

  * Full support for functions compiled with ParallelAccelerator
  * Safe caching of generated functions (eval of strings)
  * Expire cache when any function in call chain (even in other files) changes
  * Process for distributing pre-populated cache

* Continue to improve usability and debugging:

  * Trap more unsupported features earlier in pipeline (especially things that parfors can’t handle)
  * Error messages
  * Diagnostic tools for debugging and understanding performance
  * Better on-boarding for new users and contributors (revise docs, more examples)

* Begin refactoring existing features that cause common bug reports:

  * Enhance description of interfaces provided by Numba functions to give more type information
  * Convert older Numba function implementations to use public extension mechanisms
  * More unit testing and modularization of ParallelAccelerator passes

Medium Term: 2019H2
===================

* Unify dispatch of regular functions, ufuncs, and gufuncs
* Declare Numba 1.0 with stable interfaces
* Continue to improve usability and debugging (see above)
* Continue refactoring Numba internals to solve common bug reports (see above)
* JIT class review and improvement
* Improve compilation speed
* Improve memory management of Numba-allocated memory
* Better support for writing code transformation passes
* Make caching and parallel execution features opt-out instead of opt-in

  * add heuristic to determine if parfor passes will be beneficial

Long Term: 2020 and beyond
==========================

* Unify GPU backends (share more code and interfaces)
* Improve ahead of time compilation (for low powered devices)
* Improve cross language connections (C++, JVM?, Julia?, R?)

  * Call Numba from other languages,
  * Call from Numba into other languages

* Better support for "hybrid" CPU/GPU/TPU/etc programming
* Partial / deferred compilation of functions
* Foster integration of Numba into core PyData packages:

  * scipy/scikit-learn/scikit-image/pandas

* More support for efforts to put Numba into other applications (databases, etc) for compiling user-defined functions
* More support for usage of Numba as a “compiler toolkit” to create custom compilers (like HPAT, automatic differentiation of functions, etc)
* Investigate AST-based Numba frontend in addition to existing bytecode-based frontend