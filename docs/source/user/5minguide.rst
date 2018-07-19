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
* NumPy 1.10 - latest

How do I get it?
----------------
Numba is available as a `conda <https://conda.io/docs/>`_ package for the 
`Anaconda Python distribution <https://www.anaconda.com/>`_::

  $ conda install numba

Numba also has wheels available::

  $ pip install numba

Numba can also be compiled from source (`instructions <LINK>`_)

Numba is often used as a core package so its dependencies are kept to an
absolute minimum, however, extra packages can be installed as follows to provide
additional functionality:

* ``scipy`` - enables support for compiling ``numpy.linalg`` functions
* ``colorama`` - enables support for color highlighting in backtraces/error
  messages
* ``pyyaml`` - enables configuration of Numba via a YAML config file.
* ``icc_rt`` - allows the use of the Intel SVML (high performance short vector
  math library, x86_64 only).

Will Numba work for my code?
----------------------------
This depends on what your code "looks like", if your code is numerically
orientated (does a lot of math), uses NumPy a lot and/or has a lot of loops,
then Numba is often a good choice. In these examples we'll apply the most
fundamental of Numba's JIT decorators, ``@jit``, to try and speed up some
functions to demonstrate what works well and what does not.

Numba works well on code that "looks like" this::

    from numba import jit
    import numpy as np

    x = np.arange(100).reshape(10, 10)

    @jit
    def go_fast(a): # Function is compiled and runs in machine code
        trace = 0
        for i in range(a.shape[0]):   # Numba likes loops
            trace += np.tanh(a[i, i]) # Numba likes NumPy functions
        return a + trace              # Numba likes NumPy broadcasting

    print(go_fast(x))


It works reasonably well on code with well defined numerically orientated loops
(it will "lift" out the loops and compile those to machine code and run the rest
in the Python interpreter)::

    from numba import jit
    import numpy as np

    x = np.arange(20)

    @jit
    def loop_lift(x):
        a = object() # Numba runs this in the interpreter
        acc = 0
        for i in x:  # Numba will compile this loop to machine code!
            acc +=i
        return a, acc

    print(loop_lift(x))


It won't work very well, if at all, on code that "looks like" this::

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

The ``nopython`` mode *gotcha*.
-------------------------------
Pretty much all first time users are caught by this so it deserves a mention...
The Numba ``@jit`` decorator operates in two modes, ``nopython`` mode and
``object mode``. When a ``@jit`` decorated function is called, Numba will first
try and compile the function so that it will run entirely without the
involvement of the Python interpreter (like the ``go_fast`` example above).
If this succeed the function is compiled in so called ``nopython`` mode, this
gives the best performance.

Should the compilation in ``nopython`` mode fail, all is not lost, Numba will
try again at compilation in ``object mode``. In this mode Numba will identify
loops that it can compile and compile those into functions that run in machine
code, and it will run the rest of the code in the interpreter (like in the
``loop_lift`` example above). As users often want to ensure that ``nopython``
is in use, it can be supplied as a ``kwarg`` to the ``jit`` decorator
(``@jit(nopython=True)``), alternatively, for convenience, ``@njit`` obtained
via ``from numba import njit`` is a alias of this.

To help find if loops have been lifted, the numba executable ``numba`` has the
``--annotated`` option which will print diagnostic information about lifted
loops. Further, the compiled function has a method ``inspect_types()`` which if
supplied the ``kwarg`` ``pretty=True`` will produce an annotated digest if run
from a terminal or notebook. Using the above ``loop_lift`` function, try::

    print(loop_lift.inspect_types(pretty=True))


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

    @jit
    def go_fast(a): # Function is compiled and runs in machine code
        trace = 0
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

As a side note, if compilation time is an issue, Numba JIT supports `on-disk
caching <LINK>`_ of compiled functions and also has an `Ahead-Of-Time <LINK>`_
compilation mode.

How fast is it?
---------------
Assuming Numba can operate in ``nopython`` mode, or at least compile some loops,
it will target compilation to your specific CPU. Speed up varies depending on
application but can be one to two orders of magnitude. Numba has a
`performance guide <LINK>`_ that covers common options for gaining extra
performance.

How does Numba work?
--------------------
Glossing over a huge amount of detail, Numba reads the Python bytecode for a
decorated function and combines this with information about the types of the
input arguments to the function. It then forms a Numba specific internal
representation(IR) of the function and translates this to LLVM IR, it then
invokes the `LLVM compiler <https://llvm.org/>`_ machinery on the LLVM IR to
compile a machine code version of your function. This compiled version is then
used every time your function is called.

Other things of interest:
-------------------------
Numba has quite a few decorators, we've seen ``@jit`` and ``@njit``, but there's
also:

* ``@vectorize`` - produces NumPy ``ufunc`` s (with all the ``ufunc`` methods
  supported). `Docs are here <LINK>`_.
* ``@guvectorize`` - produces NumPy generalized ``ufunc`` s.
  `Docs are here <LINK>`_.
* ``@stencil`` - declare a function as a kernel for a stencil like operation.
  `Docs are here <LINK>`_.
* ``@jitclass`` - for jit aware classes. `Docs are here <LINK>`_.
* ``@cfunc`` - declare a function for use as a native call back (to be called from
  C/C++ etc). `Docs are here <LINK>`_.
* ``@overload`` - register your own implementation of a function for use in
  nopython mode, e.g. ``@overload(scipy.special.j0)``. `Docs are here <LINK>`_.

Extra options available in some decorators:

* ``parallel = True`` - enable `automatic parallelization <LINK NB PARFORS>`_ of
  the function.
* ``fastmath = True`` - enable `fast-math <LINK NB fast-math>`_ behaviour for
  the function.

ctypes/cffi/cython interoperability:

* ``cffi`` - `CFFI  <LINK NB CFFI DOCS>`_ functions are supported.
* ``ctypes`` - `ctypes  <LINK NB ctypes DOCS>`_  wrapped functions are supported
  .
* Cython exported functions `are callable <LINK NB Cython ext DOCS>`_.

GPU targets:
~~~~~~~~~~~~

Numba can target `Nvidia CUDA <https://developer.nvidia.com/cuda-zone>`_ and
(experimentally) `AMD ROC <https://rocm.github.io/>`_ GPUs. You can write a
kernel in pure Python and have Numba handle the computation and data movement
(or do this explicitly). Click for Numba documentation on `CUDA <LINK>`_ or
`ROC <LINK>`_.
