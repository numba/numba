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
cache directory (see :envvar:`NUMBA_CACHE_DIR`). The index of the cache is
stored in a ``.nbi`` file, with one index per function, and it lists all the
overloaded signatures compiled for the function. The *object code* is stored in
files with an ``.nbc`` extension, one file per overload. The data in both files
is serialized with :mod:`pickle`.

.. note:: On Python <=3.7, Numba extends ``pickle`` using the pure-Python
          pickler. To use the faster C Pickler, install ``pickle5``
          from ``pip``. ``pickle5`` backports Python 3.8 pickler features.


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

- Cache invalidation fails to recognize changes in symbols defined in a
  different file.
- Global variables are treated as constants. The cache will remember the value
  in the global variable used at compilation. On cache load, the cached
  function will not rebind to the new value of the global variable.


.. _cache-sharing:

Cache Sharing
-------------

It is safe to share and reuse the contents in the cache directory on a
different machine. The cache remembers the CPU model and the available
CPU features during compilation. If the CPU model and the CPU features do
not match exactly, the cache contents will not be considered.
(Also see :envvar:`NUMBA_CPU_NAME`)

If the cache directory is shared on a network filesystem, concurrent
read/write of the cache is safe only if file replacement operation is atomic
for the filesystem. Numba always writes to a unique temporary file first, it
then replaces the target cache file path with the temporary file. Numba is
tolerant against lost cache files and lost cache entries.

.. _cache-clearing:

Cache Clearing
--------------

The cache is invalidated when the corresponding source file is modified.
However, it is necessary sometimes to clear the cache directory manually.
For instance, changes in the compiler will not be recognized because the source
files are not modified.

To clear the cache, the cache directory can be simply removed.

Removing the cache directory when a Numba application is running may cause an
``OSError`` exception to be raised at the compilation site.

Related Environment Variables
-----------------------------

See :ref:`env-vars for caching <numba-envvars-caching>`.
