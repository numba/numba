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

.. _cache-security:

Security
--------

.. warning::

   The cache is deserialized with :mod:`pickle`, therefore loading a
   maliciously crafted cache file can execute arbitrary code. The cache
   directory must be treated as trusted: do not use a shared,
   world-writable, or otherwise untrusted location (see
   :envvar:`NUMBA_CACHE_DIR`).


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
  of the global variable at compilation time. On cache load, the cached
  function will not rebind to the new value of the global variable.


.. _cache-sharing:

Cache Sharing
-------------

It is operationally safe to share and reuse the contents of the cache
directory on a different machine. The cache remembers the CPU model and the
available CPU features during compilation. If the CPU model and the CPU
features do not match exactly, the cache contents will not be considered.
(Also see :envvar:`NUMBA_CPU_NAME`)

.. warning::

   "Safe" here is an operational property, not a security one (see
   :ref:`cache-security`).

If the cache directory is shared on a network filesystem, concurrent
read/write of the cache is safe only if file replacement operation is atomic
for the filesystem. Numba always writes to a unique temporary file first, it
then replaces the target cache file path with the temporary file. Numba is
tolerant against lost or corrupt cache files and cache entries.

.. _cache-containers:

Caching in Containers and CI
----------------------------

A common pattern is to warm the cache at build time (for example in a
``Dockerfile`` ``RUN`` step or a CI job) and reuse it at run time. A cache
miss is silent, so it helps to know what has to stay identical between the
two environments.

A cached entry is used only if all of the following match:

- **The cache location.** With the default in-tree locator the cache lives in
  ``__pycache__`` next to the source file, so it moves with the source. With
  :envvar:`NUMBA_CACHE_DIR` or the user-wide locator, the subdirectory is
  derived from the *absolute path* of the source file's directory, so the
  source must be at the same path at build and run time.
- **Writability of the cache directory.** Numba checks that the cache
  directory is writable before using it, even just to read from it. If the
  in-tree ``__pycache__`` is read-only at run time (e.g. a read-only root
  filesystem or a different user), Numba falls back to the user-wide
  directory and the warmed cache is not found.
- **The source file contents.** The index stores a hash of the source file's
  contents. File timestamps are not part of the check, so copying files
  (e.g. ``COPY`` in a ``Dockerfile``) does not invalidate the cache.
- **The Numba and Python versions**, which are part of the cache file names.
- **The platform and CPU** (see :ref:`cache-sharing`): the operating system
  and architecture (e.g. ``x86_64-unknown-linux-gnu``), the CPU model
  (e.g. ``skylake``) and the instruction sets it supports (e.g. AVX2).
  Compiled code is optimised for the CPU it was built on, so building on one
  machine and running on another with a different CPU causes a miss. Set
  :envvar:`NUMBA_CPU_NAME` to ``generic`` in both environments to produce
  portable, but potentially slower, code.
- **The function signature**, i.e. the argument types used at run time must
  have been compiled during the warm-up.

A minimal ``Dockerfile`` that satisfies these constraints::

    FROM python:3.12-slim
    ENV NUMBA_CACHE_DIR=/opt/numba_cache \
        NUMBA_CPU_NAME=generic
    RUN pip install numba
    COPY app/ /app/
    RUN python /app/warmup.py && chmod -R a+rwX /opt/numba_cache
    WORKDIR /app
    CMD ["python", "/app/main.py"]

Here ``warmup.py`` imports the application and calls each cached function
with the argument types used in production. Source files stay at ``/app`` and
the cache directory stays writable for whichever user runs the container.

To check whether the cache is used, run with :envvar:`NUMBA_DEBUG_CACHE` set
to ``1``. A hit prints ``index loaded from`` followed by ``data loaded from``
lines. A miss followed by compilation prints ``data saved to`` lines instead.

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

Cache Locators
--------------

Numba uses cache locators to determine where to store and find cached compiled
functions. The following cache locators are available:

**InTreeCacheLocator**
    Stores cache files next to the source files in a ``__pycache__`` directory,
    similar to how Python stores ``.pyc`` files.

**InTreeCacheLocatorFsAgnostic**
    Deprecated. Source freshness is now checked with a hash of the file
    contents, which is already independent of filesystem timestamps. Use
    ``InTreeCacheLocator`` instead.

**UserWideCacheLocator**
    Stores cache files in a user-wide application directory using the
    ``appdirs`` package.

**IPythonCacheLocator**
    Stores cache files in an IPython-specific cache directory.

**ZipCacheLocator**
    Handles caching for functions defined in zip files.

The order and selection of cache locators can be customized using the
:envvar:`NUMBA_CACHE_LOCATOR_CLASSES` environment variable.
