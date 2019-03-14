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

- ``setup.py`` - Standard Python distutils/setuptools script
- ``MANIFEST.in`` - Distutils packaging instructions
- ``requirements.txt`` - Pip package requirements, not used by conda
- ``versioneer.py`` - Handles automatic setting of version in installed package from git tags
- ``.flake8`` - Preferences for code formatting.  Files should be fixed and removed from exception list as time allows.
- ``buildscripts/condarecipe.local`` - Conda build recipe
- ``buildscripts/remove_unwanted_files.py`` - Helper script to clean up files.  Used by build recipes.
- ``buildscripts/condarecipe_clone_icc_rt`` - Recipe to build a standalone icc_rt package.  Not used anymore?


Continuous Integration
''''''''''''''''''''''
- ``.binstar.yml`` - Binstar Build CI config (inactive)
- ``azure-pipelines.yml`` - Azure Pipelines CI config (active: Win/Mac/Linux)
- ``buildscripts/azure/`` - Azure Pipeline configuration for specific platforms
- ``.travis.yml`` - Travis CI config (active: Mac/Linux, will be dropped in the future)
- ``appveyor.yml`` - Appveyor CI config (inactive: Win)
- ``buildscripts/appveyor/`` - Appveyor build scripts
- ``buildscripts/incremental/`` - Generic scripts for building Numba on various CI systems
- ``codecov.yml`` - Codecov.io coverage reporting


Documentation / Examples
''''''''''''''''''''''''
- ``LICENSE`` - License for Numba
- ``LICENSES.third-party`` - License for third party code vendored into Numba
- ``README.rst`` - README for repo, also uploaded to PyPI
- ``CONTRIBUTING.md`` - Documentation on how to contribute to project (out of date, should remove in favor of docs)
- ``AUTHORS`` - List of Github users who have contributed PRs (out of date)
- ``CHANGE_LOG`` - History of Numba releases, also directly embedded into Sphinx documentation
- ``docs/`` - Documentation source
- ``docs/_templates/`` - Directory for templates (to override defaults with Sphinx theme)
- ``docs/Makefile`` - Used to build Sphinx docs with ``make``
- ``docs/source`` - ReST source for Numba documentation
- ``docs/_static/`` - Static CSS and image assets for Numba docs
- ``docs/gh-pages.py`` - Utility script to update Numba docs (stored as gh-pages)
- ``docs/make.bat`` - Not used (remove?)
- ``examples/`` - Example scripts demonstrating numba (re/move to numba-examples repo?)
- ``examples/notebooks/`` - Example notebooks (re/move to numba-examples repo?)
- ``benchmarks/`` - Benchmark scripts (re/move to numba-examples repo?)
- ``tutorials/`` - Tutorial notebooks (definitely out of date, should remove and direct to external tutorials)


Numba Source Code
-----------------

Numba ships with both the source code and tests in one package.

- ``numba/`` - all of the source code and tests


Compiler Infrastructure
'''''''''''''''''''''''

These are the core components of the compiler

- ``numba/decorators.py`` - User-facing decorators for compiling regular functions on the CPU
- ``numba/compiler.py`` - Compiler pipelines and flags
- ``numba/dispatcher.py`` - Dispatcher objects are compiled functions produced by ``@jit``.  A dispatcher has different implementations for different type signatures.
- ``numba/bytecode.py`` - Bytecode parsing and function identity (??)
- ``numba/serialize.py`` - Support for pickling compiled functions
- ``numba/capsulethunk.h`` - Used by typeconv
- ``numba/typeconv`` - Implementation of type casting and type signature matching
- ``numba/typeconv/test.cpp`` - 
- ``numba/typeconv/typeconv.hpp`` - 
- ``numba/typeconv/typeconv.py`` - 
- ``numba/typeconv/__init__.py`` - 
- ``numba/typeconv/rules.py`` - 
- ``numba/typeconv/typeconv.cpp`` - 
- ``numba/typeconv/castgraph.py`` - 
- ``numba/typeconv/_typeconv.cpp`` - 
- ``numba/rewrites`` - Rewrite passes used by compiler
- ``numba/rewrites/__init__.py`` - Loads all rewrite passes so they are put into the registry
- ``numba/rewrites/registry.py`` - Registry object for collecting rewrit passes
- ``numba/rewrites/ir_print.py`` - Write print() calls into special print nodes in the IR
- ``numba/rewrites/static_raise.py`` - Convert exceptions with static arguments into special form that can be lowered
- ``numba/rewrites/macros.py`` - Generic support for macro expansion in the Numba IR
- ``numba/rewrites/static_getitem.py`` - Rewrite getitem and setitem with constant arguments to allow type inference
- ``numba/rewrites/static_binop.py`` - Rewrite binary operations (specifically ``**``) with constant arguments so faster code can be generated
- ``numba/types/`` - definition of the Numba type hierarchy, used everywhere in compiler to select implementations
- ``numba/_typeof.h`` -
- ``numba/jitclass`` - Implementation of JIT compilation of Python classes
- ``numba/funcdesc.py``
- ``numba/postproc.py``
- ``numba/transforms.py``
- ``numba/tracing.py``
- ``numba/_helperlib.c``
- ``numba/ccallback.py``
- ``numba/config.py``
- ``numba/ctypes_support.py``
- ``numba/withcontexts.py``
- ``numba/analysis.py``
- ``numba/inline_closurecall.py``
- ``numba/_helpermod.c``
- ``numba/_arraystruct.h``
- ``numba/pylowering.py`` - Lowering of code in object mode
- ``numba/_dispatcher.c``
- ``numba/lowering.py``
- ``numba/typeinfer.py``
- ``numba/_npymath_exports.c``
- ``numba/_dynfuncmod.c`` - 
- ``numba/mathnames.h`` - 
- ``numba/sigutils.py`` - 
- ``numba/numpy_support.py`` - 
- ``numba/__init__.py`` - 
- ``numba/_numba_common.h`` - 
- ``numba/ir.py`` - 
- ``numba/itanium_mangler.py`` - 
- ``numba/unittest_support.py`` - 
- ``numba/_math_c99.h`` - 
- ``numba/_dynfunc.c`` - 
- ``numba/array_analysis.py`` - 
- ``numba/consts.py`` - 
- ``numba/generators.py`` - 
- ``numba/annotations`` - 
- ``numba/annotations/type_annotations.py`` - 
- ``numba/annotations/__init__.py`` - 
- ``numba/annotations/template.html`` - 
- ``numba/_pymodule.h`` - 
- ``numba/cffi_support.py`` - 
- ``numba/interpreter.py`` - 
- ``numba/caching.py`` - 
- ``numba/utils.py`` - 
- ``numba/findlib.py`` - 
- ``numba/debuginfo.py`` - 
- ``numba/unsafe`` - 
- ``numba/unsafe/refcount.py`` - 
- ``numba/unsafe/tuple.py`` - 
- ``numba/unsafe/__init__.py`` - 
- ``numba/unsafe/ndarray.py`` - 
- ``numba/unsafe/bytes.py`` - 
- ``numba/mviewbuf.c`` - 
- ``numba/pretty_annotate.py`` - 
- ``numba/_typeof.c`` - 
- ``numba/scripts`` - 
- ``numba/scripts/generate_lower_listing.py`` - 
- ``numba/scripts/__init__.py`` - 
- ``numba/errors.py`` - 
- ``numba/dummyarray.py`` - 
- ``numba/servicelib`` - 
- ``numba/servicelib/service.py`` - 
- ``numba/servicelib/threadlocal.py`` - 
- ``numba/servicelib/__init__.py`` - 
- ``numba/_runtests.py`` - 
- ``numba/dataflow.py`` - 
- ``numba/callwrapper.py`` - 
- ``numba/_dispatcher.h`` - 
- ``numba/dictobject.py`` - 
- ``numba/datamodel`` - 
- ``numba/datamodel/models.py`` - 
- ``numba/datamodel/registry.py`` - 
- ``numba/datamodel/__init__.py`` - 
- ``numba/datamodel/packer.py`` - 
- ``numba/datamodel/testing.py`` - 
- ``numba/datamodel/manager.py`` - 
- ``numba/_dispatcherimpl.cpp`` - 
- ``numba/_math_c99.c`` - 
- ``numba/special.py`` - 
- ``numba/controlflow.py`` - 
- ``numba/macro.py`` - 
- ``numba/runtests.py`` - 
- ``numba/pythonapi.py`` - 
- ``numba/extending.py`` - 
- ``numba/npdatetime.py`` - 
- ``numba/compiler_lock.py`` - 

Misc Support
''''''''''''

- ``numba/_version.py`` - Updated by versioneer
- ``numba/runtime`` - Global singleton that manages memory allocated on the heap by Numba-compiled functions
- ``numba/ir_utils.py`` - Utility functions for working with Numba IR data structures 
- ``numba/cgutils.py`` - Utility functions for generating common code patterns in LLVM IR
- ``numba/six.py`` - Vendored subset of ``six`` package for Python 2 + 3 compatibility
- ``numba/io_support.py`` - Workaround for various names of StringIO in different Python versions (should this be in six?)
- ``numba/appdirs.py`` - Vendored package for determining application config directories on every platform


Core Python Data Structures
'''''''''''''''''''''''''''
- ``numba/_hashtable.h``
- ``numba/_hashtable.c``
- ``numba/_dictobject.h``
- ``numba/_dictobject.c``
- ``numba/unicode.py``
- ``numba/typed``
- ``numba/typed/__init__.py``
- ``numba/typed/typeddict.py``


Algorithms
''''''''''
- ``numba/_random.c``
- ``numba/_lapack.c``


ParallelAccelerator
'''''''''''''''''''

Code transformation passes that 

- ``numba/parfor.py``
- ``numba/stencil.py``
- ``numba/stencilparfor.py``


Deprecated Functionality
''''''''''''''''''''''''
- ``numba/smartarray.py`` - Experiment with an array object that has both CPU and GPU backing.  Should be removed in future.


Debugging Support
'''''''''''''''''

- ``numba/targets/gdb_hook.py`` - 
- ``numba/targets/cmdlang.gdb`` - 



Type Signatures (CPU)
'''''''''''''''''''''

Some (usually older) Numba supported functionality separates the declaration
of allowed type signatures from the definition of implementations.  This
package contains registries of type signatures that must be matched during
type inference.

- ``numba/typing`` - Type signature module
- ``numba/typing/templates.py`` - Base classes for type signature templates
- ``numba/typing/cmathdecl.py`` - Python complex math (``cmath``) module
- ``numba/typing/bufproto.py`` - Interpreting objects supporting the buffer protocol
- ``numba/typing/mathdecl.py`` - Python ``math`` module
- ``numba/typing/listdecl.py`` - Python lists
- ``numba/typing/builtins.py`` - Python builtin global functions and operators
- ``numba/typing/randomdecl.py`` - Python and NumPy ``random`` modules
- ``numba/typing/setdecl.py`` - Python sets
- ``numba/typing/npydecl.py`` - NumPy ndarray (and operators), NumPy functions
- ``numba/typing/arraydecl.py`` - Python ``array`` module
- ``numba/typing/context.py`` - Implementation of typing context (class that collects methods used in type inference)
- ``numba/typing/collections.py`` - Generic container operations and namedtuples
- ``numba/typing/ctypes_utils.py`` - Typing ctypes-wrapped function pointers
- ``numba/typing/enumdecl.py`` - Enum types
- ``numba/typing/cffi_utils.py`` - Typing of CFFI objects
- ``numba/typing/typeof.py`` - Implementation of typeof operations (maps Python object to Numba type)
- ``numba/typing/npdatetime.py`` - Datetime dtype support for NumPy arrays


Target Implementations (CPU)
''''''''''''''''''''''''''''

Implementations of Python / NumPy functions and some data models.  These
modules are responsible for generating LLVM IR during lowering.  Note that
some of these modules do not have counterparts in the typing package because
newer Numba extension APIs (like overload) allow typing and implementation to
be specified together.

- ``numba/targets`` - Implementations of compilable operations
- ``numba/targets/cpu.py`` - 
- ``numba/targets/base.py`` - 
- ``numba/targets/codegen.py`` - 
- ``numba/targets/boxing.py`` - Boxing and unboxing for most data types
- ``numba/targets/intrinsics.py`` - Utilities for converting LLVM intrinsics to other math calls
- ``numba/targets/callconv.py`` - 
- ``numba/targets/iterators.py`` - 
- ``numba/targets/hashing.py`` - 
- ``numba/targets/ufunc_db.py`` - 
- ``numba/targets/setobj.py`` - Python set type
- ``numba/targets/options.py`` - Container for options that control lowering
- ``numba/targets/printimpl.py`` - Print function
- ``numba/targets/smartarray.py`` - Smart array (deprecated)
- ``numba/targets/cmathimpl.py`` - Python complex math module
- ``numba/targets/optional.py`` - Special type representing value or ``None``
- ``numba/targets/tupleobj.py`` - Tuples (statically typed as immutable struct)
- ``numba/targets/mathimpl.py`` - Python ``math`` module
- ``numba/targets/heapq.py`` - Python ``heapq`` module
- ``numba/targets/registry.py`` - 
- ``numba/targets/imputils.py`` - 
- ``numba/targets/builtins.py`` - 
- ``numba/targets/externals.py`` - 
- ``numba/targets/quicksort.py`` - Quicksort implementation used with list and array objects
- ``numba/targets/mergesort.py`` - Mergesort implementation used with array objects
- ``numba/targets/randomimpl.py`` - Python and NumPy ``random`` modules
- ``numba/targets/npyimpl.py`` - 
- ``numba/targets/slicing.py`` - Slice objects, and index calculations used in slicing
- ``numba/targets/numbers.py`` - Numeric values (int, float, etc)
- ``numba/targets/listobj.py`` - Python lists
- ``numba/targets/fastmathpass.py`` - Rewrite pass to add fastmath attributes to function call sites and binary operations
- ``numba/targets/removerefctpass.py`` - Rewrite pass to remove unnecessary incref/decref pairs
- ``numba/targets/cffiimpl.py`` - CFFI functions
- ``numba/targets/descriptors.py`` - 
- ``numba/targets/arraymath.py`` - 
- ``numba/targets/linalg.py`` - NumPy linear algebra operations
- ``numba/targets/rangeobj.py`` - Python `range` objects
- ``numba/targets/npyfuncs.py`` - Kernels used in generating some NumPy ufuncs
- ``numba/targets/arrayobj.py`` - Array operations (both NumPy and buffer protocol)
- ``numba/targets/enumimpl.py`` - Enum objects
- ``numba/targets/polynomial.py`` - ``numpy.roots`` function
- ``numba/targets/npdatetime.py`` - NumPy datetime operations




Ufunc Compiler and Runtime
''''''''''''''''''''''''''

- ``numba/npyufunc``
- ``numba/npyufunc/_internal.h``
- ``numba/npyufunc/deviceufunc.py``
- ``numba/npyufunc/gufunc_scheduler.h``
- ``numba/npyufunc/dufunc.py``
- ``numba/npyufunc/ufuncbuilder.py``
- ``numba/npyufunc/sigparse.py``
- ``numba/npyufunc/omppool.cpp``
- ``numba/npyufunc/parfor.py``
- ``numba/npyufunc/workqueue.h``
- ``numba/npyufunc/__init__.py``
- ``numba/npyufunc/array_exprs.py``
- ``numba/npyufunc/gufunc_scheduler.cpp``
- ``numba/npyufunc/_internal.c``
- ``numba/npyufunc/wrappers.py``
- ``numba/npyufunc/_ufunc.c``
- ``numba/npyufunc/workqueue.c``
- ``numba/npyufunc/tbbpool.cpp``
- ``numba/npyufunc/parallel.py``
- ``numba/npyufunc/decorators.py``



Unit Tests (CPU)
''''''''''''''''

CPU unit tests (GPU target unit tests listed in later sections

- ``runtests.py`` - Convenience script that launches test runner and turns on full compiler tracebacks
- ``run_coverage.py`` - Runs test suite with coverage tracking enabled
- ``.coveragerc`` - Coverage.py configuration
- ``numba/tests/test_*`` - Test cases
- ``numba/tests/*_usecases.py`` - Python functions compiled by some unit tests
- ``numba/tests/support.py`` - Helper functions for testin and special TestCase implementation
- ``numba/tests/dummy_module.py`` - Module used in ``test_dispatcher.py``
- ``numba/tests/npyufunc`` - ufunc / gufunc compiler tests
- ``numba/testing`` - 
- ``numba/testing/ddt.py`` - 
- ``numba/testing/__init__.py`` - 
- ``numba/testing/loader.py`` - 
- ``numba/testing/notebook.py`` - 
- ``numba/testing/main.py`` - 
- ``numba/testing/__main__.py`` - 


Command Line Utilities
''''''''''''''''''''''
- ``bin/numba`` - Command line stub, delegates to main in ``numba_entry.py``
- ``numba/numba_entry.py`` - Main function for ``numba`` command line tool
- ``numba/pycc`` - Ahead of time compilation of functions to shared library extension
- ``numba/pycc/__init__.py`` - Main function for ``pycc`` command line tool
- ``numba/pycc/cc.py`` - User-facing API for tagging functions to compile ahead of time
- ``numba/pycc/compiler.py`` - Compiler pipeline for creating standalone Python extension modules
- ``numba/pycc/llvm_types.py`` - Aliases to LLVM data types used by ``compiler.py``
- ``numba/pycc/pycc`` - Stub to call main function.  Is this still used?
- ``numba/pycc/modulemixin.c`` - C file compiled into every compiled extension.  Pulls in C source from Numba core that is needed to make extension standalone
- ``numba/pycc/platform.py`` - Portable interface to platform-specific compiler toolchains
- ``numba/pycc/decorators.py`` - Deprecated decorators for tagging functions to compile.  Use ``cc.py`` instead.


CUDA GPU Target
'''''''''''''''

Note that the CUDA target does reuse some parts of the CPU target.

- ``numba/cuda/`` - The implementation of the CUDA (NVIDIA GPU) target and associated unit tests
- ``numba/cuda/decorators.py`` - Compiler decorators for CUDA
- ``numba/cuda/dispatcher.py`` - Dispatcher for CUDA JIT functions
- ``numba/cuda/printimpl.py`` - Special implementation of device printing
- ``numba/cuda/libdevice.py`` - Registers libdevice functions
- ``numba/cuda/kernels/`` - Custom kernels for reduction and transpose 
- ``numba/cuda/device_init.py`` - Initializes the CUDA target when imported
- ``numba/cuda/compiler.py`` - Compiler pipeline for CUDA target
- ``numba/cuda/intrinsic_wrapper.py`` - CUDA device intrinsics (shuffle, ballot, etc)
- ``numba/cuda/initialize.py`` - Defered initialization of the CUDA device and subsystem.  Called only when user imports ``numba.cuda``
- ``numba/cuda/simulator_init.py`` - Initalizes the CUDA simulator subsystem (only when user requests it with env var)
- ``numba/cuda/random.py`` - Implementation of random number generator
- ``numba/cuda/api.py`` - User facing APIs imported into ``numba.cuda.*``
- ``numba/cuda/stubs.py`` - Python placeholders for functions that only can be used in GPU device code
- ``numba/cuda/simulator/`` - Simulate execution of CUDA kernels in Python interpreter
- ``numba/cuda/vectorizers.py`` - Subclasses of ufunc/gufunc compilers for CUDA
- ``numba/cuda/args.py`` - 
- ``numba/cuda/target.py`` - 
- ``numba/cuda/cudamath.py`` - 
- ``numba/cuda/errors.py`` - 
- ``numba/cuda/cudaimpl.py`` - 
- ``numba/cuda/nvvmutils.py`` - 
- ``numba/cuda/testing.py`` - 
- ``numba/cuda/cudadecl.py`` - 
- ``numba/cuda/codegen.py`` - 
- ``numba/cuda/cudadrv/`` - Wrapper around CUDA driver API
- ``numba/cuda/tests/`` - CUDA unit tests, skipped when CUDA is not detected
- ``numba/cuda/tests/cudasim/`` - Tests of CUDA simulator
- ``numba/cuda/tests/nocuda/`` - Tests for NVVM functionality when CUDA not present
- ``numba/cuda/tests/cudapy/`` - Tests of compiling Python functions for GPU
- ``numba/cuda/tests/cudadrv/`` - Tests of Python wrapper around CUDA API


ROCm GPU Target
'''''''''''''''

Note that the ROCm target does reuse some parts of the CPU target, but
duplicates some code from CUDA target.  A future refactoring could pull out
the common subset of CUDA and ROCm.  An older version of this target was based
on the HSA API, so "hsa" appears in many places.

- ``numba/roc`` - ROCm GPU target for AMD GPUs
- ``numba/roc/descriptor.py``
- ``numba/roc/enums.py``
- ``numba/roc/mathdecl.py``
- ``numba/roc/compiler.py``
- ``numba/roc/mathimpl.py``
- ``numba/roc/hlc``
- ``numba/roc/hlc/config.py``
- ``numba/roc/hlc/__init__.py``
- ``numba/roc/hlc/common.py``
- ``numba/roc/hlc/hlc.py``
- ``numba/roc/hlc/libhlc.py``
- ``numba/roc/initialize.py``
- ``numba/roc/hsadecl.py``
- ``numba/roc/dispatch.py``
- ``numba/roc/hsaimpl.py``
- ``numba/roc/__init__.py``
- ``numba/roc/README.md``
- ``numba/roc/api.py``
- ``numba/roc/gcn_occupancy.py``
- ``numba/roc/stubs.py``
- ``numba/roc/vectorizers.py``
- ``numba/roc/target.py``
- ``numba/roc/hsadrv`` - Python wrapper around ROCm (based on HSA) driver API calls
- ``numba/roc/codegen.py``
- ``numba/roc/decorators.py``
- ``numba/roc/tests`` - Unit tests for ROCm target
- ``numba/roc/tests/hsapy`` - Tests of compiling ROCm kernels written in Python syntax
- ``numba/roc/tests/hsadrv`` - Tests of Python wrapper on platform API.



