.. _roadmap:

*******************
Numba Roadmap
*******************

This document describes features we want in numba, but do not
have yet. We will first list what we want in upcoming versions,
and then what features we want in general. Those features can
always be added to the roadmap for upcoming versions if someone
is interested in implementing them.

Numba Versions
==============

1.0
---
What we want for 1.0 is:

    * Numba loader (loader_)
    * IR stages (stages_)
    * More robust type inferencer
    * Well-defined runtime

        - including exception support

    * Debug info
    * numba --annotate tool (annotate_)
    * parallel tasks (green threads, typed channels, scheduler)
    * generators on top of the green thread model

We also like some minimal Cython support, in addition to the longer
term goals of SEP 200. One idea from Zaur Shibzukhov is to provide
support for Cython pxd overlays::

    # foo.py

    def my_function(a):
        b = 2
        return a ** b

Such a module can be overlain with a Cython pxd file, e.g.

.. code-block:: cython

    # foo.pxd

    cimport cython

    @cython.locals(b=double)
    cpdef my_function(double a)

For some inspiration of what we can do with pxd overlays, see also:
https://github.com/cython/cython/blob/master/Cython/Compiler/FlowControl.pxd

We can now compile ``foo.py`` with Cython. We should be able to similarly
compile ``foo.py`` with numba, using ``pycc`` as well as at runtime to produce
a new module with annotated functions compiled in the right order.

Thing we want
=============
We will order these from less involved to more involved,
to provide different entry points to numba development.

Less intricate
==============
Here as some less intricate topics, providing easier starting points
for new contributors:

NumPy Type Inference
--------------------
Full/more support for type inference on NumPy functions:

    * http://numba.pydata.org/numba-doc/dev/doc/type_inference.html
    * https://github.com/numba/numba/tree/devel/numba/type_inference/modules

Typed Containers
----------------
We currently have (naive implementations of):

    * typedlist  (https://github.com/numba/numba/blob/devel/numba/containers/typedlist.py)
    * typedtuple (https://github.com/numba/numba/blob/devel/numba/containers/typedtuple.py)

But we want many more! Some ideas:

    * typeddict
    * typedset
    * typedchannel

        - one thread-safe (nogil) and one requiring the GIL

Perhaps also the ordered variants of ``typeddict`` and ``typedset``.

Intrinsics
----------
Support for LLVM intrinsics (we only have instructions at the moment):

    * http://numba.pydata.org/numba-doc/dev/doc/interface_c.html#using-intrinsics
    * https://github.com/numba/numba/blob/devel/numba/intrinsic/numba_intrinsic.py

E.g.::

    intrin = numba.declare_intrinsic(int64(), "llvm.readcyclecounter")
    print intrin()

.. _annotate:

Source Annotator
----------------
Analogous to ``cython --annotate``, a tool that annotates numba source code
and finds and highlights which parts contain object calls. Ideally, this would
also include, for each source line (expand on click?):

    * The final (unoptimized) LLVM bitcode

        - And optionally the optimized code and/or assembly

    * Code from intermediate numba representations

        - After we start implementing several layers of IR,
          see http://numba.pydata.org/numba-doc/dev/doc/ir.html

    * The type of each sub-expression and variable (on hover?)

Issue: https://github.com/numba/numba/issues/105

.. _loader:

Numba Loader
------------
Allow two forms of code caching:

    * For distribution (portable IR)
    * Locally on disk (unportable compiled binaries)

The first bullet will allow library writers to distribute
numba code while not being tied to numba versions that users
have installed. This would be similar to distribution of
C code compiled from Cython source:

.. code-block:: bash

    $ numba --compile foo.py
    Writing foo.numba

We can now distribute ``foo.numba``.
Load code explicitly:

.. code-block:: python

    from numba import loader
    foo = loader.load("foo.numba")
    foo.func()

... or use an import hook:

.. code-block:: python

    from numba import loader
    loader.install_hook()

    import foo
    foo.func()

... or compile to extension modules during setup:

.. code-block:: python

    from numba.loader import NumbaExtension

    setup(
        ...,
        ext_modules=[
            NumbaExtension("foo.bar",
                           sources=["foo/bar.numba"]),
            ],
    )

Or perhaps more conveniently, implement ``find_numba_modules()``
to find all ``*.numba`` source files and return a list of
``NumbaExtension``.

This also plays into the IR discussion found here:
http://numba.pydata.org/numba-doc/dev/doc/ir.html

JIT Special Methods
-------------------
Jit operations that result in calls to special methods like ``__len__``,
``__getitem__``, etc. This requires some careful thought as to the stage where
this transformation should take place.

Array Expressions
-----------------
Array Expression support in Numba, including scans, reductions, etc.
Or maybe we should make Blaze a hard dependency for that?

More intricate
==============
More intricate topics, in no particular order:

Extension Types
---------------
* Support autojit class inheritance
* Support partial method specialization

::

    @Any(int_, Any)
    def my_method(self, a, b):
        ...

Infer the return type and specialize on parameter type ``b``, but
fix parameter type ``a``.

* Allow annotation of pure-python only methods (don't compile)

What we also need is native dispatch of foreign callables, in a
sustainable way: SEP 200 and SEP 201
    * https://github.com/numfocus/sep/
    * Widen support in scientific community

Recursion
---------
Support recursion for autojit functions and methods:

    * Construct call graph
    * Build condensation graph and resolve

        - similar to cycles in SSA

Exceptions
----------
Support for zero-cost exceptions: support in the runtime libraries for
all models:

    * True zero-cost exceptions

        - Stack trace through libunwind/apple backtrace/LLVM info
          based on instruction pointer
        - http://llvm.org/docs/LangRef.html#invoke-instruction
        - http://llvm.org/docs/ExceptionHandling.html

    * Setjmp/longjmp

        - Optionally with exception analysis to allow cheap cleanup for
          the simpler cases

    * Costful exceptions

        - "return -1"
        - Implement fast ``NumbaErr_Occurred()`` or change calling
          convention for native or void returns

We also need to allow users to take the pointer to a numba ``jit``
function::

    numba.addressof(my_numba_function)

We can allow specifying an exception model:

    * ``propagate=False``: This does not propagate, but uses
      PyErr_WriteUnraisable

    * ``propagate=True``: Implies ``write_unraisable=False``. Callers
      check with ``NumbaErr_Occurred()`` (or for NULL if object return).
      Maybe also specify a range of badvals:

        - int -> 0xdeadbeef (``ret == 0xdeadbeef && NumbaErr_Occurred()``)
        - float -> float('nan') (``ret != ret && NumbaErr_Occurred()``)

.. NOTE:: We have ``numba.addressof()``, but we don't have ``NumbaErr_Occurred()``
          yet.

Debug info
----------
GDB Backtraces!

See:

    * https://github.com/llvmpy/llvmpy/blob/debuginfo/llvm/debuginfo.py
    * https://github.com/llvmpy/llvmpy/blob/debuginfo/test/test_debuginfo.py

Or is there a successor to that?

Struct references
-----------------
Use cheap heap allocated objects + garbage collection?

    * or atomic reference counts?

Use stack-allocation + escape analysis?

Blaze
-----
Blaze support:

    * compile abstract blaze expressions into kernels
    * generate native call to blaze kernel

Generators/parallel Tasks
-------------------------
Support for generators based on green threading support:

    * Write typed channels as autojit class
    * Support green thread context switching
    * Rewrite iteration over generators

::

    def g(N):
        for i in range(N):
            yield f(i)      # write to channel (triggering a context switch)

    def consume():
        gen = g(100)        # create task with bound parameter N and channel C
        for i in gen:       # read from C until exhaustion
            use(i)

See also
https://groups.google.com/a/continuum.io/forum/#!searchin/numba-users/generators/numba-users/gaVgArRrXqw/HTyTzaXsW_EJ
for how this compares to generators based on closures.

Python 3.3 support
------------------
We support Python 3.3, but we can additionally support type-annotations:

.. code-block:: python3

    def func(a: int_, b: float_) -> double:
        ...

Maybe this can work with numba.automodule(my_numba_module) as well as with
jit and autojit methods.


GPUs
----

    * SPIR support (OpenCL)

Vector support
--------------

    * Vector-types in Numba

        - What does this look like?

