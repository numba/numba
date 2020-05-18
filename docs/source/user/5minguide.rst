.. _numba-5_mins:

A ~5 minute guide to Numba
==========================

Numba is a just-in-time compiler for Python that works best on code that uses
NumPy arrays and functions, and loops. The most common way to use Numba is
through its collection of decorators that can be applied to your functions to
instruct Numba to compile them. When a call is made to a Numba decorated
function it is compiled to machine code "just-in-time" for execution and all or
part of your code can subsequently run at native machine code speed!

Out of the box Numba works with the following:

* OS: Windows (32 and 64 bit), OSX and Linux (32 and 64 bit)
* Architecture: x86, x86_64, ppc64le. Experimental on armv7l, armv8l (aarch64).
* GPUs: Nvidia CUDA. Experimental on AMD ROC.
* CPython
* NumPy 1.15 - latest

How do I get it?
----------------
Numba is available as a `conda <https://conda.io/docs/>`_ package for the
`Anaconda Python distribution <https://www.anaconda.com/>`_::

  $ conda install numba

Numba also has wheels available::

  $ pip install numba

Numba can also be
:ref:`compiled from source <numba-source-install-instructions>`, although we do
not recommend it for first-time Numba users.

Numba is often used as a core package so its dependencies are kept to an
absolute minimum, however, extra packages can be installed as follows to provide
additional functionality:

* ``scipy`` - enables support for compiling ``numpy.linalg`` functions.
* ``colorama`` - enables support for color highlighting in backtraces/error
  messages.
* ``pyyaml`` - enables configuration of Numba via a YAML config file.
* ``icc_rt`` - allows the use of the Intel SVML (high performance short vector
  math library, x86_64 only). Installation instructions are in the
  :ref:`performance tips <intel-svml>`.

Will Numba work for my code?
----------------------------
This depends on what your code looks like, if your code is numerically
orientated (does a lot of math), uses NumPy a lot and/or has a lot of loops,
then Numba is often a good choice. In these examples we'll apply the most
fundamental of Numba's JIT decorators, ``@jit``, to try and speed up some
functions to demonstrate what works well and what does not.

Numba works well on code that looks like this::

    from numba import jit
    import numpy as np

    x = np.arange(100).reshape(10, 10)

    @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
    def go_fast(a): # Function is compiled to machine code when called the first time
        trace = 0.0
        for i in range(a.shape[0]):   # Numba likes loops
            trace += np.tanh(a[i, i]) # Numba likes NumPy functions
        return a + trace              # Numba likes NumPy broadcasting

    print(go_fast(x))


It won't work very well, if at all, on code that looks like this::

    from numba import jit
    import pandas as pd

    x = {'a': [1, 2, 3], 'b': [20, 30, 40]}

    @jit
    def use_pandas(a): # Function will not benefit from Numba jit
        df = pd.DataFrame.from_dict(a) # Numba doesn't know about pd.DataFrame
        df += 1                        # Numba doesn't understand what this is
        return df.cov()                # or this!

    print(use_pandas(x))

Note that Pandas is not understood by Numba and as a result Numba would simply
run this code via the interpreter but with the added cost of the Numba internal
overheads!

What is ``nopython`` mode?
--------------------------
The Numba ``@jit`` decorator fundamentally operates in two compilation modes,
``nopython`` mode and ``object`` mode. In the ``go_fast`` example above,
``nopython=True`` is set in the ``@jit`` decorator, this is instructing Numba to
operate in ``nopython`` mode. The behaviour of the ``nopython`` compilation mode
is to essentially compile the decorated function so that it will run entirely
without the involvement of the Python interpreter. This is the recommended and
best-practice way to use the Numba ``jit`` decorator as it leads to the best
performance.

Should the compilation in ``nopython`` mode fail, Numba can compile using
``object mode``, this is a fall back mode for the ``@jit`` decorator if
``nopython=True`` is not set (as seen in the ``use_pandas`` example above). In
this mode Numba will identify loops that it can compile and compile those into
functions that run in machine code, and it will run the rest of the code in the
interpreter. For best performance avoid using this mode!

How to measure the performance of Numba?
----------------------------------------
First, recall that Numba has to compile your function for the argument types
given before it executes the machine code version of your function, this takes
time. However, once the compilation has taken place Numba caches the machine
code version of your function for the particular types of arguments presented.
If it is called again the with same types, it can reuse the cached version
instead of having to compile again.

A really common mistake when measuring performance is to not account for the
above behaviour and to time code once with a simple timer that includes the
time taken to compile your function in the execution time.

For example::

    from numba import jit
    import numpy as np
    import time

    x = np.arange(100).reshape(10, 10)

    @jit(nopython=True)
    def go_fast(a): # Function is compiled and runs in machine code
        trace = 0.0
        for i in range(a.shape[0]):
            trace += np.tanh(a[i, i])
        return a + trace

    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    start = time.time()
    go_fast(x)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    start = time.time()
    go_fast(x)
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))

This, for example prints::

    Elapsed (with compilation) = 0.33030009269714355
    Elapsed (after compilation) = 6.67572021484375e-06

A good way to measure the impact Numba JIT has on your code is to time execution
using the `timeit <https://docs.python.org/3/library/timeit.html>`_ module
functions, these measure multiple iterations of execution and, as a result,
can be made to accommodate for the compilation time in the first execution.

As a side note, if compilation time is an issue, Numba JIT supports
:ref:`on-disk caching <jit-decorator-cache>` of compiled functions and also has
an :ref:`Ahead-Of-Time <aot-compilation>` compilation mode.

How fast is it?
---------------
Assuming Numba can operate in ``nopython`` mode, or at least compile some loops,
it will target compilation to your specific CPU. Speed up varies depending on
application but can be one to two orders of magnitude. Numba has a
:ref:`performance guide <performance-tips>` that covers common options for
gaining extra performance.

How does Numba work?
--------------------
Numba reads the Python bytecode for a decorated function and combines this with
information about the types of the input arguments to the function. It analyzes
and optimizes your code, and finally uses the LLVM compiler library to generate
a machine code version of your function, tailored to your CPU capabilities. This
compiled version is then used every time your function is called.

Other things of interest:
-------------------------
Numba has quite a few decorators, we've seen ``@jit``, but there's
also:

* ``@njit`` - this is an alias for ``@jit(nopython=True)`` as it is so commonly
  used!
* ``@vectorize`` - produces NumPy ``ufunc`` s (with all the ``ufunc`` methods
  supported). :ref:`Docs are here <vectorize>`.
* ``@guvectorize`` - produces NumPy generalized ``ufunc`` s.
  :ref:`Docs are here <guvectorize>`.
* ``@stencil`` - declare a function as a kernel for a stencil like operation.
  :ref:`Docs are here <numba-stencil>`.
* ``@jitclass`` - for jit aware classes. :ref:`Docs are here <jitclass>`.
* ``@cfunc`` - declare a function for use as a native call back (to be called
  from C/C++ etc). :ref:`Docs are here <cfunc>`.
* ``@overload`` - register your own implementation of a function for use in
  nopython mode, e.g. ``@overload(scipy.special.j0)``.
  :ref:`Docs are here <high-level-extending>`.

Extra options available in some decorators:

* ``parallel = True`` - :ref:`enable <jit-decorator-parallel>` the
  :ref:`automatic parallelization <numba-parallel>` of the function.
* ``fastmath = True`` - enable :ref:`fast-math <jit-decorator-fastmath>`
  behaviour for the function.

ctypes/cffi/cython interoperability:

* ``cffi`` - The calling of :ref:`CFFI  <cffi-support>` functions is supported
  in ``nopython`` mode.
* ``ctypes`` - The calling of :ref:`ctypes  <ctypes-support>` wrapped
  functions is supported in ``nopython`` mode.
  .
* Cython exported functions :ref:`are callable <cython-support>`.

GPU targets:
~~~~~~~~~~~~

Numba can target `Nvidia CUDA <https://developer.nvidia.com/cuda-zone>`_ and
(experimentally) `AMD ROC <https://rocm.github.io/>`_ GPUs. You can write a
kernel in pure Python and have Numba handle the computation and data movement
(or do this explicitly). Click for Numba documentation on
:ref:`CUDA <cuda-index>` or :ref:`ROC <roc-index>`.
