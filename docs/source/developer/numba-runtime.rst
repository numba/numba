.. _arch-numba-runtime:

======================
Notes on Numba Runtime
======================


The *Numba Runtime (NRT)* provides the language runtime to the *nopython mode*
Python subset.  NRT is a standalone C library with a Python binding.  This
allows NPM runtime feature to be used without the GIL.  Currently, the only
language feature implemented in NRT is memory management.


Memory Management
=================

NRT implements memory management for NPM code.  It uses *atomic reference count*
for threadsafe, deterministic memory management.  NRT maintains a separate
``MemInfo`` structure for storing information about each allocation.

Cooperating with CPython
------------------------

For NRT to cooperate with CPython, the NRT python binding provides adaptors for
converting python objects that export a memory region.  When such an
object is used as an argument to a NPM function, a new ``MemInfo`` is created
and it acquires a reference to the Python object.  When a NPM value is returned
to the Python interpreter, the associated ``MemInfo`` (if any) is checked.  If
the ``MemInfo`` references a Python object, the underlying Python object is
released and returned instead.  Otherwise, the ``MemInfo`` is wrapped in a
Python object and returned.  Additional process maybe required depending on
the type.

The current implementation supports Numpy array and any buffer-exporting types.


Compiler-side Cooperation
-------------------------

NRT reference counting requires the compiler to emit incref/decref operations
according to the usage.  When the reference count drops to zero, the compiler
must call the destructor routine in NRT.


.. _nrt-refct-opt-pass:

Optimizations
-------------

The compiler is allowed to emit incref/decref operations naively.  It relies
on an optimization pass that to remove the redundant reference count
operations.

The optimization pass runs on block level to avoid control flow analysis.
It depends on LLVM function optimization pass to simplify the control flow,
stack-to-register, and simplify instructions.  It works by matching and
removing incref and decref pairs within each block.


Quirks
------

Since the `refcount optimization pass <nrt-refct-opt-pass_>`_ requires LLVM
function optimization pass, the pass works on the LLVM IR as text.  The
optimized IR is then materialized again as a new LLVM in-memory bitcode object.


Debugging Leaks
---------------

To debug reference leaks in NRT MemInfo, each MemInfo python object has a
``.refcount`` attribute for inspection.  To get the MemInfo from a ndarray
allocated by NRT, use the ``.base`` attribute.

To debug memory leaks in NRT, the ``numba.runtime.rtsys`` defines
``.get_allocation_stats()``.  It returns a namedtuple containing the
number of allocation and deallocation since the start of the program.
Checking that the allocation and deallocation counters are matching is the
simplest way to know if the NRT is leaking.


Debugging Leaks in C
--------------------

The start of `numba/runtime/nrt.h <https://github.com/numba/numba/blob/master/numba/runtime/nrt.h>`_
has these lines:

.. code-block:: C

  /* Debugging facilities - enabled at compile-time */
  /* #undef NDEBUG */
  #if 0
  #   define NRT_Debug(X) X
  #else
  #   define NRT_Debug(X) if (0) { X; }
  #endif

Undefining NDEBUG (uncomment the ``#undef NDEBUG`` line) enables the assertion
check in NRT.

Enabling the NRT_Debug (replace ``#if 0`` with ``#if 1``) turns on
debug print inside NRT.

Future Plan
===========

The plan for NRT is to make a standalone shared library that can be linked to
Numba compiled code, including use within the Python interpreter and without
the Python interpreter.  To make that work, we will be doing some refactoring:

* numba NPM code references statically compiled code in "helperlib.c".  Those
  functions should be moved to NRT.
