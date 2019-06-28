.. _developer-caching:

================
Notes on Caching
================

Numba supports caching of compiled functions into the filesystem for future
use of the same functions.


The Implementation
==================

Caching is done by saving the compiled *object code*, the ELF object of the
executable code.  By using the *object code*, cached functions have minimal
overhead because no compilation is needed. The cached data is saved under the
``__pycache__`` directory. The index of the cache is stored in a ``.nbi``
file, with one index per function, and it lists all the overloaded signatures
compiled for the function. The *object code* is stored in files with an
``.nbc`` extension, one file per overload.


Requirements for Cacheability
-----------------------------

Developers should note the requirements of a function to permit it to be cached
to ensure that the features they are working on are compatible with caching.

Requirements for cacheable function:

- The LLVM module must be *self-contained*, meaning that it cannot rely on
  other compiled units without linking to them.
- The only allowed external symbols are from the
  :ref:`NRT <arch-numba-runtime>` or other common symbols from system libraries
  (i.e. libc and libm).

Debugging note:

- Look for the usage of ``inttoptr`` in the LLVM IR or
  ``target_context.add_dynamic_add()`` in the lowering code in Python.
  They indicate potential usage of runtime address. Not all uses are
  problematic and some are necessary. Only the conversion of constant integers
  into pointers will affect caching.
- Misuse of dynamic address or dynamic symbols will likely result in a
  segfault.
- Linking order matters because unused symbols are dropped after linking.
  Linking should start from the leaf nodes of the dependency graph.


Features Compatible with Caching
--------------------------------

The following features are explicitly verified to work with caching.

- ufuncs and gufuncs for the ``cpu`` and ``parallel`` target
- parallel accelerator features (i.e. ``parallel=True``)


Caching Limitations
-------------------

This is a list of known limitation of the cache:

- Functions using ``hash(str)`` will produce unexpected results when loaded
  from cache. This also affects dictionary usage (i.e. ``numba.typed.Dict``).
- Cache invalidation fails to recognize changes in symbols defined in a
  different file.
