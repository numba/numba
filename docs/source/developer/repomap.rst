A Map of the Numba Repository
=============================

The Numba repository is quite large, and due to age has functionality spread
around many locations.  To help orient developers, this document will try to
summarize where different categories of functionality can be found.

.. note::
    It is likely that the organization of the code base will change in the
    future to improve organization.  Follow `issue #3807 <https://github.com/numba/numba/issues/3807>`_
    for more details.


Support Files
-------------

Build and Packaging
'''''''''''''''''''

- :ghfile:`setup.py` - Standard Python distutils/setuptools script
- :ghfile:`MANIFEST.in` - Distutils packaging instructions
- :ghfile:`requirements.txt` - Pip package requirements, not used by conda
- :ghfile:`versioneer.py` - Handles automatic setting of version in
  installed package from git tags
- :ghfile:`.flake8` - Preferences for code formatting.  Files should be
  fixed and removed from the exception list as time allows.
- :ghfile:`buildscripts/condarecipe.local` - Conda build recipe
- :ghfile:`buildscripts/remove_unwanted_files.py` - Helper script to remove
  files that will not compile under Python 2. Used by build recipes.
- :ghfile:`buildscripts/condarecipe_clone_icc_rt` - Recipe to build a
  standalone icc_rt package.


Continuous Integration
''''''''''''''''''''''
- :ghfile:`.binstar.yml` - Binstar Build CI config (inactive)
- :ghfile:`azure-pipelines.yml` - Azure Pipelines CI config (active:
  Win/Mac/Linux)
- :ghfile:`buildscripts/azure/` - Azure Pipeline configuration for specific
  platforms
- :ghfile:`.travis.yml` - Travis CI config (active: Mac/Linux, will be
  dropped in the future)
- :ghfile:`buildscripts/appveyor/` - Appveyor build scripts
- :ghfile:`buildscripts/incremental/` - Generic scripts for building Numba
  on various CI systems
- :ghfile:`codecov.yml` - Codecov.io coverage reporting


Documentation / Examples
''''''''''''''''''''''''
- :ghfile:`LICENSE` - License for Numba
- :ghfile:`LICENSES.third-party` - License for third party code vendored
  into Numba
- :ghfile:`README.rst` - README for repo, also uploaded to PyPI
- :ghfile:`CONTRIBUTING.md` - Documentation on how to contribute to project
  (out of date, should be updated to point to Sphinx docs)
- :ghfile:`AUTHORS` - List of Github users who have contributed PRs (out of
  date)
- :ghfile:`CHANGE_LOG` - History of Numba releases, also directly embedded
  into Sphinx documentation
- :ghfile:`docs/` - Documentation source
- :ghfile:`docs/_templates/` - Directory for templates (to override defaults
  with Sphinx theme)
- :ghfile:`docs/Makefile` - Used to build Sphinx docs with ``make``
- :ghfile:`docs/source` - ReST source for Numba documentation
- :ghfile:`docs/_static/` - Static CSS and image assets for Numba docs
- :ghfile:`docs/gh-pages.py` - Utility script to update Numba docs (stored
  as gh-pages)
- :ghfile:`docs/make.bat` - Not used (remove?)
- :ghfile:`examples/` - Example scripts demonstrating numba (re/move to
  numba-examples repo?)
- :ghfile:`examples/notebooks/` - Example notebooks (re/move to
  numba-examples repo?)
- :ghfile:`benchmarks/` - Benchmark scripts (re/move to numba-examples
  repo?)
- :ghfile:`tutorials/` - Tutorial notebooks (definitely out of date, should
  remove and direct to external tutorials)
- :ghfile:`numba/scripts/generate_lower_listing.py` - Dump all registered
  implementations decorated with ``@lower*`` for reference
  documentation.  Currently misses implementations from the higher
  level extension API.



Numba Source Code
-----------------

Numba ships with both the source code and tests in one package.

- :ghfile:`numba/` - all of the source code and tests


Public API
''''''''''

These define aspects of the public Numba interface.

- :ghfile:`numba/decorators.py` - User-facing decorators for compiling
  regular functions on the CPU
- :ghfile:`numba/extending.py` - Public decorators for extending Numba
  (``overload``, ``intrinsic``, etc)
- :ghfile:`numba/ccallback.py` - ``@cfunc`` decorator for compiling
  functions to a fixed C signature.  Used to make callbacks.
- :ghfile:`numba/npyufunc/decorators.py` - ufunc/gufunc compilation
  decorators
- :ghfile:`numba/config.py` - Numba global config options and environment
  variable handling
- :ghfile:`numba/annotations` - Gathering and printing type annotations of
  Numba IR
- :ghfile:`numba/pretty_annotate.py` - Code highlighting of Numba functions
  and types (both ANSI terminal and HTML)


Dispatching
'''''''''''

- :ghfile:`numba/dispatcher.py` - Dispatcher objects are compiled functions
  produced by ``@jit``.  A dispatcher has different implementations
  for different type signatures.
- :ghfile:`numba/_dispatcher.{h,c}` - C interface to C++ dispatcher
  implementation
- :ghfile:`numba/_dispatcherimpl.cpp` - C++ dispatcher implementation (for
  speed on common data types)


Compiler Pipeline
'''''''''''''''''

- :ghfile:`numba/compiler.py` - Compiler pipelines and flags
- :ghfile:`numba/errors.py` - Numba exception and warning classes
- :ghfile:`numba/ir.py` - Numba IR data structure objects
- :ghfile:`numba/bytecode.py` - Bytecode parsing and function identity (??)
- :ghfile:`numba/interpreter.py` - Translate Python interpreter bytecode to
  Numba IR
- :ghfile:`numba/analysis.py` - Utility functions to analyze Numba IR
  (variable lifetime, prune branches, etc)
- :ghfile:`numba/dataflow.py` - Dataflow analysis for Python bytecode (used
  in analysis.py)
- :ghfile:`numba/controlflow.py` - Control flow analysis of Numba IR and
  Python bytecode
- :ghfile:`numba/typeinfer.py` - Type inference algorithm
- :ghfile:`numba/transforms.py` - Numba IR transformations
- :ghfile:`numba/rewrites` - Rewrite passes used by compiler
- :ghfile:`numba/rewrites/__init__.py` - Loads all rewrite passes so they
  are put into the registry
- :ghfile:`numba/rewrites/registry.py` - Registry object for collecting
  rewrite passes
- :ghfile:`numba/rewrites/ir_print.py` - Write print() calls into special
  print nodes in the IR
- :ghfile:`numba/rewrites/static_raise.py` - Converts exceptions with static
  arguments into a special form that can be lowered
- :ghfile:`numba/rewrites/macros.py` - Generic support for macro expansion
  in the Numba IR
- :ghfile:`numba/rewrites/static_getitem.py` - Rewrites getitem and setitem
  with constant arguments to allow type inference
- :ghfile:`numba/rewrites/static_binop.py` - Rewrites binary operations
  (specifically ``**``) with constant arguments so faster code can be
  generated
- :ghfile:`numba/inline_closurecall.py` - Inlines body of closure functions
  to call site.  Support for array comprehensions, reduction inlining,
  and stencil inlining.
- :ghfile:`numba/macro.py` - Alias to ``numba.rewrites.macros``
- :ghfile:`numba/postproc.py` - Postprocessor for Numba IR that computes
  variable lifetime, inserts del operations, and handles generators
- :ghfile:`numba/lowering.py` - General implementation of lowering Numba IR
  to LLVM
- :ghfile:`numba/withcontexts.py` - General scaffolding for implementing
  context managers in nopython mode, and the objectmode context
  manager
- :ghfile:`numba/pylowering.py` - Lowering of Numba IR in object mode
- :ghfile:`numba/pythonapi.py` - LLVM IR code generation to interface with
  CPython API


Type Management
'''''''''''''''

- :ghfile:`numba/typeconv/` - Implementation of type casting and type
  signature matching in both C++ and Python
- :ghfile:`numba/capsulethunk.h` - Used by typeconv
- :ghfile:`numba/types/` - definition of the Numba type hierarchy, used
  everywhere in compiler to select implementations
- :ghfile:`numba/consts.py` - Constant inference (used to make constant
  values available during codegen when possible)
- :ghfile:`numba/datamodel` - LLVM IR representations of data types in
  different contexts
- :ghfile:`numba/datamodel/models.py` - Models for most standard types
- :ghfile:`numba/datamodel/registry.py` - Decorator to register new data
  models
- :ghfile:`numba/datamodel/packer.py` - Pack typed values into a data
  structure
- :ghfile:`numba/datamodel/testing.py` - Data model tests (this should
  move??)
- :ghfile:`numba/datamodel/manager.py` - Map types to data models


Compiled Extensions
'''''''''''''''''''

Numba uses a small amount of compiled C/C++ code for core
functionality, like dispatching and type matching where performance
matters, and it is more convenient to encapsulate direct interaction
with CPython APIs.

- :ghfile:`numba/_arraystruct.h` - Struct for holding NumPy array
  attributes.  Used in helperlib and the Numba Runtime.
- :ghfile:`numba/_helperlib.c` - C functions required by Numba compiled code
  at runtime.  Linked into ahead-of-time compiled modules
- :ghfile:`numba/_helpermod.c` - Python extension module with pointers to
  functions from ``_helperlib.c`` and ``_npymath_exports.c``
- :ghfile:`numba/_npymath_exports.c` - Export function pointer table to
  NumPy C math functions
- :ghfile:`numba/_dynfuncmod.c` - Python extension module exporting
  _dynfunc.c functionality
- :ghfile:`numba/_dynfunc.c` - C level Environment and Closure objects (keep
  in sync with numba/target/base.py)
- :ghfile:`numba/mathnames.h` - Macros for defining names of math functions
- :ghfile:`numba/_pymodule.h` - C macros for Python 2/3 portable naming of C
  API functions
- :ghfile:`numba/_math_c99.{h,c}` - C99 math compatibility (needed Python
  2.7 on Windows, compiled with VS2008)
- :ghfile:`numba/mviewbuf.c` - Handles Python memoryviews
- :ghfile:`numba/_typeof.{h,c}` - C implementation of type fingerprinting,
  used by dispatcher
- :ghfile:`numba/_numba_common.h` - Portable C macro for marking symbols
  that can be shared between object files, but not outside the
  library.



Misc Support
''''''''''''

- :ghfile:`numba/_version.py` - Updated by versioneer
- :ghfile:`numba/runtime` - Language runtime.  Currently manages
  reference-counted memory allocated on the heap by Numba-compiled
  functions
- :ghfile:`numba/ir_utils.py` - Utility functions for working with Numba IR
  data structures
- :ghfile:`numba/cgutils.py` - Utility functions for generating common code
  patterns in LLVM IR
- :ghfile:`numba/six.py` - Vendored subset of ``six`` package for Python 2 +
  3 compatibility
- :ghfile:`numba/io_support.py` - Workaround for various names of StringIO
  in different Python versions (should this be in six?)
- :ghfile:`numba/utils.py` - Python 2 backports of Python 3 functionality
  (also imports local copy of ``six``)
- :ghfile:`numba/appdirs.py` - Vendored package for determining application
  config directories on every platform
- :ghfile:`numba/compiler_lock.py` - Global compiler lock because Numba's usage
  of LLVM is not thread-safe
- :ghfile:`numba/special.py` - Python stub implementations of special Numba
  functions (prange, gdb*)
- :ghfile:`numba/servicelib/threadlocal.py` - Thread-local stack used by GPU
  targets
- :ghfile:`numba/servicelib/service.py` - Should be removed?
- :ghfile:`numba/itanium_mangler.py` - Python implementation of Itanium C++
  name mangling
- :ghfile:`numba/findlib.py` - Helper function for locating shared libraries
  on all platforms
- :ghfile:`numba/debuginfo.py` - Helper functions to construct LLVM IR debug
  info
- :ghfile:`numba/unsafe` - ``@intrinsic`` helper functions that can be used
  to implement direct memory/pointer manipulation from nopython mode
  functions
- :ghfile:`numba/unsafe/refcount.py` - Read reference count of object
- :ghfile:`numba/unsafe/tuple.py` - Replace a value in a tuple slot
- :ghfile:`numba/unsafe/ndarray.py` - NumPy array helpers
- :ghfile:`numba/unsafe/bytes.py` - Copying and dereferencing data from void
  pointers
- :ghfile:`numba/dummyarray.py` - Used by GPU backends to hold array information
  on the host, but not the data.
- :ghfile:`numba/callwrapper.py` - Handles argument unboxing and releasing
  the GIL when moving from Python to nopython mode
- :ghfile:`numba/ctypes_support.py` - Import this instead of ``ctypes`` to
  workaround portability issue with Python 2.7
- :ghfile:`numba/cffi_support.py` - Alias of numba.typing.cffi_utils for
  backward compatibility (still needed?)
- :ghfile:`numba/numpy_support.py` - Helper functions for working with NumPy
  and translating Numba types to and from NumPy dtypes.
- :ghfile:`numba/tracing.py` - Decorator for tracing Python calls and
  emitting log messages
- :ghfile:`numba/funcdesc.py` - Classes for describing function metadata
  (used in the compiler)
- :ghfile:`numba/sigutils.py` - Helper functions for parsing and normalizing
  Numba type signatures
- :ghfile:`numba/serialize.py` - Support for pickling compiled functions
- :ghfile:`numba/caching.py` - Disk cache for compiled functions
- :ghfile:`numba/npdatetime.py` - Helper functions for implementing NumPy
  datetime64 support


Core Python Data Types
''''''''''''''''''''''

- :ghfile:`numba/_hashtable.{h,c}` - Adaptation of the Python 3.7 hash table
  implementation
- :ghfile:`numba/cext/dictobject.{h,c}` - C level implementation of typed
  dictionary
- :ghfile:`numba/dictobject.py` - Nopython mode wrapper for typed dictionary
- :ghfile:`numba/cext/listobject.{h,c}` - C level implementation of typed list
- :ghfile:`numba/listobject.py` - Nopython mode wrapper for typed list
- :ghfile:`numba/typedobjectutils.py` - Common utilities for typed dictionary
  and list
- :ghfile:`numba/unicode.py` - Unicode strings (Python 3.5 and later)
- :ghfile:`numba/typed` - Python interfaces to statically typed containers
- :ghfile:`numba/typed/typeddict.py` - Python interface to typed dictionary
- :ghfile:`numba/typed/typedlist.py` - Python interface to typed list
- :ghfile:`numba/jitclass` - Implementation of JIT compilation of Python
  classes
- :ghfile:`numba/generators.py` - Support for lowering Python generators


Math
''''

- :ghfile:`numba/_random.c` - Reimplementation of NumPy / CPython random
  number generator
- :ghfile:`numba/_lapack.c` - Wrappers for calling BLAS and LAPACK functions
  (requires SciPy)


ParallelAccelerator
'''''''''''''''''''

Code transformation passes that extract parallelizable code from
a function and convert it into multithreaded gufunc calls.

- :ghfile:`numba/parfor.py` - General ParallelAccelerator
- :ghfile:`numba/stencil.py` - Stencil function decorator (implemented
  without ParallelAccelerator)
- :ghfile:`numba/stencilparfor.py` - ParallelAccelerator implementation of
  stencil
- :ghfile:`numba/array_analysis.py` - Array analysis passes used in
  ParallelAccelerator


Debugging Support
'''''''''''''''''

- :ghfile:`numba/targets/gdb_hook.py` - Hooks to jump into GDB from nopython
  mode
- :ghfile:`numba/targets/cmdlang.gdb` - Commands to setup GDB for setting
  explicit breakpoints from Python


Type Signatures (CPU)
'''''''''''''''''''''

Some (usually older) Numba supported functionality separates the
declaration of allowed type signatures from the definition of
implementations.  This package contains registries of type signatures
that must be matched during type inference.

- :ghfile:`numba/typing` - Type signature module
- :ghfile:`numba/typing/templates.py` - Base classes for type signature
  templates
- :ghfile:`numba/typing/cmathdecl.py` - Python complex math (``cmath``)
  module
- :ghfile:`numba/typing/bufproto.py` - Interpreting objects supporting the
  buffer protocol
- :ghfile:`numba/typing/mathdecl.py` - Python ``math`` module
- :ghfile:`numba/typing/listdecl.py` - Python lists
- :ghfile:`numba/typing/builtins.py` - Python builtin global functions and
  operators
- :ghfile:`numba/typing/randomdecl.py` - Python and NumPy ``random`` modules
- :ghfile:`numba/typing/setdecl.py` - Python sets
- :ghfile:`numba/typing/npydecl.py` - NumPy ndarray (and operators), NumPy
  functions
- :ghfile:`numba/typing/arraydecl.py` - Python ``array`` module
- :ghfile:`numba/typing/context.py` - Implementation of typing context
  (class that collects methods used in type inference)
- :ghfile:`numba/typing/collections.py` - Generic container operations and
  namedtuples
- :ghfile:`numba/typing/ctypes_utils.py` - Typing ctypes-wrapped function
  pointers
- :ghfile:`numba/typing/enumdecl.py` - Enum types
- :ghfile:`numba/typing/cffi_utils.py` - Typing of CFFI objects
- :ghfile:`numba/typing/typeof.py` - Implementation of typeof operations
  (maps Python object to Numba type)
- :ghfile:`numba/typing/npdatetime.py` - Datetime dtype support for NumPy
  arrays


Target Implementations (CPU)
''''''''''''''''''''''''''''

Implementations of Python / NumPy functions and some data models.
These modules are responsible for generating LLVM IR during lowering.
Note that some of these modules do not have counterparts in the typing
package because newer Numba extension APIs (like overload) allow
typing and implementation to be specified together.

- :ghfile:`numba/targets` - Implementations of compilable operations
- :ghfile:`numba/targets/cpu.py` - Context for code gen on CPU
- :ghfile:`numba/targets/base.py` - Base class for all target contexts
- :ghfile:`numba/targets/codegen.py` - Driver for code generation
- :ghfile:`numba/targets/boxing.py` - Boxing and unboxing for most data
  types
- :ghfile:`numba/targets/intrinsics.py` - Utilities for converting LLVM
  intrinsics to other math calls
- :ghfile:`numba/targets/callconv.py` - Implements different calling
  conventions for Numba-compiled functions
- :ghfile:`numba/targets/iterators.py` - Iterable data types and iterators
- :ghfile:`numba/targets/hashing.py` - Hashing algorithms
- :ghfile:`numba/targets/ufunc_db.py` - Big table mapping types to ufunc
  implementations
- :ghfile:`numba/targets/setobj.py` - Python set type
- :ghfile:`numba/targets/options.py` - Container for options that control
  lowering
- :ghfile:`numba/targets/printimpl.py` - Print function
- :ghfile:`numba/targets/cmathimpl.py` - Python complex math module
- :ghfile:`numba/targets/optional.py` - Special type representing value or
  ``None``
- :ghfile:`numba/targets/tupleobj.py` - Tuples (statically typed as
  immutable struct)
- :ghfile:`numba/targets/mathimpl.py` - Python ``math`` module
- :ghfile:`numba/targets/heapq.py` - Python ``heapq`` module
- :ghfile:`numba/targets/registry.py` - Registry object for collecting
  implementations for a specific target
- :ghfile:`numba/targets/imputils.py` - Helper functions for lowering
- :ghfile:`numba/targets/builtins.py` - Python builtin functions and
  operators
- :ghfile:`numba/targets/externals.py` - Registers external C functions
  needed to link generated code
- :ghfile:`numba/targets/quicksort.py` - Quicksort implementation used with
  list and array objects
- :ghfile:`numba/targets/mergesort.py` - Mergesort implementation used with
  array objects
- :ghfile:`numba/targets/randomimpl.py` - Python and NumPy ``random``
  modules
- :ghfile:`numba/targets/npyimpl.py` - Implementations of most NumPy ufuncs
- :ghfile:`numba/targets/slicing.py` - Slice objects, and index calculations
  used in slicing
- :ghfile:`numba/targets/numbers.py` - Numeric values (int, float, etc)
- :ghfile:`numba/targets/listobj.py` - Python lists
- :ghfile:`numba/targets/fastmathpass.py` - Rewrite pass to add fastmath
  attributes to function call sites and binary operations
- :ghfile:`numba/targets/removerefctpass.py` - Rewrite pass to remove
  unnecessary incref/decref pairs
- :ghfile:`numba/targets/cffiimpl.py` - CFFI functions
- :ghfile:`numba/targets/descriptors.py` - empty base class for all target
  descriptors (is this needed?)
- :ghfile:`numba/targets/arraymath.py` - Math operations on arrays (both
  Python and NumPy)
- :ghfile:`numba/targets/linalg.py` - NumPy linear algebra operations
- :ghfile:`numba/targets/rangeobj.py` - Python `range` objects
- :ghfile:`numba/targets/npyfuncs.py` - Kernels used in generating some
  NumPy ufuncs
- :ghfile:`numba/targets/arrayobj.py` - Array operations (both NumPy and
  buffer protocol)
- :ghfile:`numba/targets/enumimpl.py` - Enum objects
- :ghfile:`numba/targets/polynomial.py` - ``numpy.roots`` function
- :ghfile:`numba/targets/npdatetime.py` - NumPy datetime operations


Ufunc Compiler and Runtime
''''''''''''''''''''''''''

- :ghfile:`numba/npyufunc` - ufunc compiler implementation
- :ghfile:`numba/npyufunc/_internal.{h,c}` - Python extension module with
  helper functions that use CPython & NumPy C API
- :ghfile:`numba/npyufunc/_ufunc.c` - Used by `_internal.c`
- :ghfile:`numba/npyufunc/deviceufunc.py` - Custom ufunc dispatch for
  non-CPU targets
- :ghfile:`numba/npyufunc/gufunc_scheduler.{h,cpp}` - Schedule work chunks
  to threads
- :ghfile:`numba/npyufunc/dufunc.py` - Special ufunc that can compile new
  implementations at call time
- :ghfile:`numba/npyufunc/ufuncbuilder.py` - Top-level orchestration of
  ufunc/gufunc compiler pipeline
- :ghfile:`numba/npyufunc/sigparse.py` - Parser for generalized ufunc
  indexing signatures
- :ghfile:`numba/npyufunc/parfor.py` - gufunc lowering for
  ParallelAccelerator
- :ghfile:`numba/npyufunc/parallel.py` - Codegen for ``parallel`` target
- :ghfile:`numba/npyufunc/array_exprs.py` - Rewrite pass for turning array
  expressions in regular functions into ufuncs
- :ghfile:`numba/npyufunc/wrappers.py` - Wrap scalar function kernel with
  loops
- :ghfile:`numba/npyufunc/workqueue.{h,c}` - Threading backend based on
  pthreads/Windows threads and queues
- :ghfile:`numba/npyufunc/omppool.cpp` - Threading backend based on OpenMP
- :ghfile:`numba/npyufunc/tbbpool.cpp` - Threading backend based on TBB



Unit Tests (CPU)
''''''''''''''''

CPU unit tests (GPU target unit tests listed in later sections

- :ghfile:`runtests.py` - Convenience script that launches test runner and
  turns on full compiler tracebacks
- :ghfile:`run_coverage.py` - Runs test suite with coverage tracking enabled
- :ghfile:`.coveragerc` - Coverage.py configuration
- :ghfile:`numba/runtests.py` - Entry point to unittest runner
- :ghfile:`numba/_runtests.py` - Implementation of custom test runner
  command line interface
- :ghfile:`numba/tests/test_*` - Test cases
- :ghfile:`numba/tests/*_usecases.py` - Python functions compiled by some
  unit tests
- :ghfile:`numba/tests/support.py` - Helper functions for testing and
  special TestCase implementation
- :ghfile:`numba/tests/dummy_module.py` - Module used in
  ``test_dispatcher.py``
- :ghfile:`numba/tests/npyufunc` - ufunc / gufunc compiler tests
- :ghfile:`numba/unittest_support.py` - Import instead of unittest to handle
  portability issues
- :ghfile:`numba/testing` - Support code for testing
- :ghfile:`numba/testing/ddt.py` - decorators for test cases
- :ghfile:`numba/testing/loader.py` - Find tests on disk
- :ghfile:`numba/testing/notebook.py` - Support for testing notebooks
- :ghfile:`numba/testing/main.py` - Numba test runner


Command Line Utilities
''''''''''''''''''''''
- :ghfile:`bin/numba` - Command line stub, delegates to main in
  ``numba_entry.py``
- :ghfile:`numba/numba_entry.py` - Main function for ``numba`` command line
  tool
- :ghfile:`numba/pycc` - Ahead of time compilation of functions to shared
  library extension
- :ghfile:`numba/pycc/__init__.py` - Main function for ``pycc`` command line
  tool
- :ghfile:`numba/pycc/cc.py` - User-facing API for tagging functions to
  compile ahead of time
- :ghfile:`numba/pycc/compiler.py` - Compiler pipeline for creating
  standalone Python extension modules
- :ghfile:`numba/pycc/llvm_types.py` - Aliases to LLVM data types used by
  ``compiler.py``
- :ghfile:`numba/pycc/pycc` - Stub to call main function.  Is this still
  used?
- :ghfile:`numba/pycc/modulemixin.c` - C file compiled into every compiled
  extension.  Pulls in C source from Numba core that is needed to make
  extension standalone
- :ghfile:`numba/pycc/platform.py` - Portable interface to platform-specific
  compiler toolchains
- :ghfile:`numba/pycc/decorators.py` - Deprecated decorators for tagging
  functions to compile.  Use ``cc.py`` instead.


CUDA GPU Target
'''''''''''''''

Note that the CUDA target does reuse some parts of the CPU target.

- :ghfile:`numba/cuda/` - The implementation of the CUDA (NVIDIA GPU) target
  and associated unit tests
- :ghfile:`numba/cuda/decorators.py` - Compiler decorators for CUDA kernels
  and device functions
- :ghfile:`numba/cuda/dispatcher.py` - Dispatcher for CUDA JIT functions
- :ghfile:`numba/cuda/printimpl.py` - Special implementation of device printing
- :ghfile:`numba/cuda/libdevice.py` - Registers libdevice functions
- :ghfile:`numba/cuda/kernels/` - Custom kernels for reduction and transpose
- :ghfile:`numba/cuda/device_init.py` - Initializes the CUDA target when
  imported
- :ghfile:`numba/cuda/compiler.py` - Compiler pipeline for CUDA target
- :ghfile:`numba/cuda/intrinsic_wrapper.py` - CUDA device intrinsics
  (shuffle, ballot, etc)
- :ghfile:`numba/cuda/initialize.py` - Defered initialization of the CUDA
  device and subsystem.  Called only when user imports ``numba.cuda``
- :ghfile:`numba/cuda/simulator_init.py` - Initalizes the CUDA simulator
  subsystem (only when user requests it with env var)
- :ghfile:`numba/cuda/random.py` - Implementation of random number generator
- :ghfile:`numba/cuda/api.py` - User facing APIs imported into ``numba.cuda.*``
- :ghfile:`numba/cuda/stubs.py` - Python placeholders for functions that
  only can be used in GPU device code
- :ghfile:`numba/cuda/simulator/` - Simulate execution of CUDA kernels in
  Python interpreter
- :ghfile:`numba/cuda/vectorizers.py` - Subclasses of ufunc/gufunc compilers
  for CUDA
- :ghfile:`numba/cuda/args.py` - Management of kernel arguments, including
  host<->device transfers
- :ghfile:`numba/cuda/target.py` - Typing and target contexts for GPU
- :ghfile:`numba/cuda/cudamath.py` - Type signatures for math functions in
  CUDA Python
- :ghfile:`numba/cuda/errors.py` - Validation of kernel launch configuration
- :ghfile:`numba/cuda/nvvmutils.py` - Helper functions for generating
  NVVM-specific IR
- :ghfile:`numba/cuda/testing.py` - Support code for creating CUDA unit
  tests and capturing standard out
- :ghfile:`numba/cuda/cudadecl.py` - Type signatures of CUDA API (threadIdx,
  blockIdx, atomics) in Python on GPU
- :ghfile:`numba/cuda/cudaimpl.py` - Implementations of CUDA API functions
  on GPU
- :ghfile:`numba/cuda/codegen.py` - Code generator object for CUDA target
- :ghfile:`numba/cuda/cudadrv/` - Wrapper around CUDA driver API
- :ghfile:`numba/cuda/tests/` - CUDA unit tests, skipped when CUDA is not
  detected
- :ghfile:`numba/cuda/tests/cudasim/` - Tests of CUDA simulator
- :ghfile:`numba/cuda/tests/nocuda/` - Tests for NVVM functionality when
  CUDA not present
- :ghfile:`numba/cuda/tests/cudapy/` - Tests of compiling Python functions
  for GPU
- :ghfile:`numba/cuda/tests/cudadrv/` - Tests of Python wrapper around CUDA
  API


ROCm GPU Target
'''''''''''''''

Note that the ROCm target does reuse some parts of the CPU target, and
duplicates some code from CUDA target.  A future refactoring could
pull out the common subset of CUDA and ROCm.  An older version of this
target was based on the HSA API, so "hsa" appears in many places.

- :ghfile:`numba/roc` - ROCm GPU target for AMD GPUs
- :ghfile:`numba/roc/descriptor.py` - TargetDescriptor subclass for ROCm
  target
- :ghfile:`numba/roc/enums.py` - Internal constants
- :ghfile:`numba/roc/mathdecl.py` - Declarations of math functions that can
  be used on device
- :ghfile:`numba/roc/mathimpl.py` - Implementations of math functions for
  device
- :ghfile:`numba/roc/compiler.py` - Compiler pipeline for ROCm target
- :ghfile:`numba/roc/hlc` - Wrapper around LLVM interface for AMD GPU
- :ghfile:`numba/roc/initialize.py` - Register ROCm target for ufunc/gufunc
  compiler
- :ghfile:`numba/roc/hsadecl.py` - Type signatures for ROCm device API in
  Python
- :ghfile:`numba/roc/hsaimpl.py` - Implementations of ROCm device API
- :ghfile:`numba/roc/dispatch.py` - ufunc/gufunc dispatcher
- :ghfile:`numba/roc/README.md` - Notes on testing target (should be
  deleted)
- :ghfile:`numba/roc/api.py` - Host API for ROCm actions
- :ghfile:`numba/roc/gcn_occupancy.py` - Heuristic to compute occupancy of
  kernels
- :ghfile:`numba/roc/stubs.py` - Host stubs for device functions
- :ghfile:`numba/roc/vectorizers.py` - Builds ufuncs
- :ghfile:`numba/roc/target.py` - Target and typing contexts
- :ghfile:`numba/roc/hsadrv` - Python wrapper around ROCm (based on HSA)
  driver API calls
- :ghfile:`numba/roc/codegen.py` - Codegen subclass for ROCm target
- :ghfile:`numba/roc/decorators.py` - ``@jit`` decorator for kernels and
  device functions
- :ghfile:`numba/roc/tests` - Unit tests for ROCm target
- :ghfile:`numba/roc/tests/hsapy` - Tests of compiling ROCm kernels written
  in Python syntax
- :ghfile:`numba/roc/tests/hsadrv` - Tests of Python wrapper on platform
  API.
