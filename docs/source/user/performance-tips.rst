Performance Tips
================

This is a short guide to features present in Numba that can help with obtaining
the best performance from code. Two examples are used, both are entirely
contrived and exist purely for pedagogical reasons to motivate discussion.
The first is the computation of the trigonometric identity
``cos(x)^2 + sin(x)^2``, the second is a simple element wise square root of a
vector with reduction over summation. All performance numbers are indicative
only and unless otherwise stated were taken from running on an Intel ``i7-4790``
CPU (4 hardware threads) with an input of ``np.arange(1.e7)``.

.. note::
   A reasonably effective approach to achieving high performance code is to
   profile the code running with real data and use that to guide performance
   tuning. The information presented here is to demonstrate features, not to act
   as canonical guidance!

No Python mode vs Object mode
-----------------------------

A common pattern is to decorate functions with ``@jit`` as this is the most
flexible decorator offered by Numba. ``@jit`` essentially encompasses two modes
of compilation, first it will try and compile the decorated function in no
Python mode, if this fails it will try again to compile the function using
object mode. Whilst the use of looplifting in object mode can enable some
performance increase, getting functions to compile under no python mode is
really the key to good performance. To make it such that only no python mode is
used and if compilation fails an exception is raised the decorators ``@njit``
and ``@jit(nopython=True)`` can be used (the first is an alias of the
second for convenience).

Loops
-----
Whilst NumPy has developed a strong idiom around the use of vector operations,
Numba is perfectly happy with loops too. For users familiar with C or Fortran,
writing Python in this style will work fine in Numba (after all, LLVM gets a
lot of use in compiling C lineage languages). For example::

    @njit
    def ident_np(x):
        return np.cos(x) ** 2 + np.sin(x) ** 2

    @njit
    def ident_loops(x):
        r = np.empty_like(x)
        n = len(x)
        for i in range(n):
            r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
        return r

The above run at almost identical speeds when decorated with ``@njit``, without
the decorator the vectorized function is a couple of orders of magnitude faster.

+-----------------+-------+----------------+
| Function Name   | @njit | Execution time |
+=================+=======+================+
| ``ident_np``    | No    |     0.581s     |
+-----------------+-------+----------------+
| ``ident_np``    | Yes   |     0.659s     |
+-----------------+-------+----------------+
| ``ident_loops`` | No    |     25.2s      |
+-----------------+-------+----------------+
| ``ident_loops`` | Yes   |     0.670s     |
+-----------------+-------+----------------+


Fastmath
--------
In certain classes of applications strict IEEE 754 compliance is less
important. As a result it is possible to relax some numerical rigour with
view of gaining additional performance. The way to achieve this behaviour in
Numba is through the use of the ``fastmath`` keyword argument::

    @njit(fastmath=False)
    def do_sum(A):
        acc = 0.
        # without fastmath, this loop must accumulate in strict order
        for x in A:
            acc += np.sqrt(x)
        return acc

    @njit(fastmath=True)
    def do_sum_fast(A):
        acc = 0.
        # with fastmath, the reduction can be vectorized as floating point
        # reassociation is permitted.
        for x in A:
            acc += np.sqrt(x)
        return acc


+-----------------+-----------------+
| Function Name   | Execution time  |
+=================+=================+
| ``do_sum``      |      35.2 ms    |
+-----------------+-----------------+
| ``do_sum_fast`` |      17.8 ms    |
+-----------------+-----------------+


Parallel=True
-------------
If code contains operations that are parallelisable (:ref:`and supported
<numba-parallel-supported>`) Numba can compile a version of that will run in
parallel on multiple native threads (no GIL!). This parallelisation is performed
automatically and is enabled by simply adding the ``parallel`` keyword
argument::

    @njit(parallel=True)
    def ident_parallel(A):
        return np.cos(x) ** 2 + np.sin(x) ** 2


Executions times are as follows:

+--------------------+-----------------+
| Function Name      | Execution time  |
+====================+=================+
| ``ident_parallel`` | 112 ms          |
+--------------------+-----------------+


The execution speed of this function with ``parallel=True`` present is
approximately 5x that of the NumPy equivalent and 6x that of standard
``@njit``.


Numba parallel execution also has support for explicit parallel loop
declaration similar to that in OpenMP. To indicate that a loop should be
executed in parallel the ``numba.prange`` function should be used, this function
behaves like Python ``range`` and if ``parallel=True`` is not set it acts
simply as an alias of ``range``. Loops induced with ``prange`` can be used for
embarrassingly parallel computation and also reductions.

Revisiting the reduce over sum example, assuming it is safe for the sum to be
accumulated out of order, the loop in ``n`` can be parallelised through the use
of ``prange``. Further, the ``fastmath=True`` keyword argument can be added
without concern in this case as the assumption that out of order execution is
valid has already been made through the use of ``parallel=True`` (as each thread
computes a partial sum).
::

    @njit(parallel=True)
    def do_sum_parallel(A):
        # each thread can accumulate its own partial sum, and then a cross
        # thread reduction is performed to obtain the result to return
        n = len(A)
        acc = 0.
        for i in prange(n):
            acc += np.sqrt(A[i])
        return acc

    @njit(parallel=True, fastmath=True)
    def do_sum_parallel_fast(A):
        n = len(A)
        acc = 0.
        for i in prange(n):
            acc += np.sqrt(A[i])
        return acc


Execution times are as follows, ``fastmath`` again improves performance.

+-------------------------+-----------------+
| Function Name           | Execution time  |
+=========================+=================+
| ``do_sum_parallel``     |      9.81 ms    |
+-------------------------+-----------------+
| ``do_sum_parallel_fast``|      5.37 ms    |
+-------------------------+-----------------+

Intel SVML
----------

Intel provides a short vector math library (SVML) that contains a large number
of optimised transcendental functions available for use as compiler
intrinsics. If the ``icc_rt`` package is present in the environment (or the SVML
libraries are simply locatable!) then Numba automatically configures the LLVM
back end to use the SVML intrinsic functions where ever possible. SVML provides
both high and low accuracy versions of each intrinsic and the version that is
used is determined through the use of the ``fastmath`` keyword. The default is
to use high accuracy which is accurate to within ``1 ULP``, however if 
``fastmath`` is set to ``True`` then the lower accuracy versions of the
intrinsics are used (answers to within ``4 ULP``).


First obtain SVML, using conda for example::

    conda install -c numba icc_rt

Rerunning the identity function example ``ident_np`` from above with various
combinations of options to ``@njit`` and with/without SVML yields the following
performance results (input size ``np.arange(1.e8)``). For reference, with just
NumPy the function executed in ``5.84s``:

+-----------------------------------+--------+-------------------+
| ``@njit`` kwargs                  |  SVML  | Execution time    |
+===================================+========+===================+
| ``None``                          | No     | 5.95s             |
+-----------------------------------+--------+-------------------+
| ``None``                          | Yes    | 2.26s             |
+-----------------------------------+--------+-------------------+
| ``fastmath=True``                 | No     | 5.97s             |
+-----------------------------------+--------+-------------------+
| ``fastmath=True``                 | Yes    | 1.8s              |
+-----------------------------------+--------+-------------------+
| ``parallel=True``                 | No     | 1.36s             |
+-----------------------------------+--------+-------------------+
| ``parallel=True``                 | Yes    | 0.624s            |
+-----------------------------------+--------+-------------------+
| ``parallel=True, fastmath=True``  | No     | 1.32s             |
+-----------------------------------+--------+-------------------+
| ``parallel=True, fastmath=True``  | Yes    | 0.576s            |
+-----------------------------------+--------+-------------------+

It is evident that SVML significantly increases the performance of this
function. The impact of ``fastmath`` in the case of SVML not being present is
zero, this is expected as there is nothing in the original function that would
benefit from relaxing numerical strictness.

Linear algebra
--------------
Numba supports most of ``numpy.linalg`` in no Python mode. The internal
implementation relies on a LAPACK and BLAS library to do the numerical work
and it obtains the bindings for the necessary functions from SciPy. Therefore,
to achieve good performance in ``numpy.linalg`` functions with Numba it is
necessary to use a SciPy built against a well optimised LAPACK/BLAS library.
In the case of the Anaconda distribution SciPy is built against Intel's MKL
which is highly optimised and as a result Numba makes use of this performance.
