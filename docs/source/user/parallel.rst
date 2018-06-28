.. Copyright (c) 2017 Intel Corporation
   SPDX-License-Identifier: BSD-2-Clause

.. _numba-parallel:

=======================================
Automatic parallelization with ``@jit``
=======================================

Setting the :ref:`parallel_jit_option` option for :func:`~numba.jit` enables
an experimental Numba feature that attempts to automatically parallelize and
perform other optimizations on (part of) a function. At the moment, this
feature only works on CPUs.

Some operations inside a user defined function, e.g., adding a scalar value to
an array, are known to have parallel semantics.  A user program may contain
many such operations and while each operation could be parallelized
individually, such an approach often has lackluster performance due to poor
cache behavior.  Instead, with auto-parallelization, Numba attempts to
identify such operations in a user program, and fuse adjacent ones together,
to form one or more kernels that are automatically run in parallel.
The process is fully automated without modifications to the user program,
which is in contrast to Numba's :func:`~numba.vectorize` or
:func:`~numba.guvectorize` mechanism, where manual effort is required
to create parallel kernels.

.. _numba-parallel-supported:

Supported Operations
====================

In this section, we give a list of all the array operations that have
parallel semantics and for which we attempt to parallelize.

#. All numba array operations that are supported by :ref:`case-study-array-expressions`,
   which include common arithmetic functions between Numpy arrays, and between
   arrays and scalars, as well as Numpy ufuncs. They are often called
   `element-wise` or `point-wise` array operations:

    * unary operators: ``+`` ``-`` ``~``
    * binary operators: ``+`` ``-`` ``*`` ``/`` ``/?`` ``%`` ``|`` ``>>`` ``^`` ``<<`` ``&`` ``**`` ``//``
    * comparison operators: ``==`` ``!=`` ``<`` ``<=`` ``>`` ``>=``
    * :ref:`Numpy ufuncs <supported_ufuncs>` that are supported in :term:`nopython mode`.
    * User defined :class:`~numba.DUFunc` through :func:`~numba.vectorize`.

#. Numpy reduction functions ``sum``, ``prod``, ``min``, ``max``, ``argmin``,
   and ``argmax``. Also, array math functions ``mean``, ``var``, and ``std``.

#. Numpy array creation functions ``zeros``, ``ones``, ``arange``, ``linspace``,
   and several random functions (rand, randn, ranf, random_sample, sample,
   random, standard_normal, chisquare, weibull, power, geometric, exponential,
   poisson, rayleigh, normal, uniform, beta, binomial, f, gamma, lognormal,
   laplace, randint, triangular).

#. Numpy ``dot`` function between a matrix and a vector, or two vectors.
   In all other cases, Numba's default implementation is used.

#. Multi-dimensional arrays are also supported for the above operations
   when operands have matching dimension and size. The full semantics of
   Numpy broadcast between arrays with mixed dimensionality or size is
   not supported, nor is the reduction across a selected dimension.

#. Array assignment in which the target is an array selection using a slice
   or a boolean array, and the value being assigned is either a scalar or
   another selection where the slice range or bitarray are inferred to be
   compatible.

#. The ``reduce`` operator of ``functools`` is supported for specifying parallel
   reductions on 1D Numpy arrays but the initial value argument is mandatory.

.. _numba-prange:

Explicit Parallel Loops
========================

Another experimental feature of this module is support for explicit parallel
loops. One can use Numba's ``prange`` instead of ``range`` to specify that a
loop can be parallelized. The user is required to make sure that the loop does
not have cross iteration dependencies except the supported reductions.

A reductions is inferred automatically if a variable is updated by a binary
function/operator using its previous value in the loop body. The initial value
of the reduction is inferred automatically for ``+=`` and ``*=`` operators.
For other functions/operators, the reduction variable should hold the initial
value right before entering the ``prange`` loop.

The example below demonstrates a parallel loop with a
reduction (``A`` is a one-dimensional Numpy array)::

    from numba import njit, prange
    @njit(parallel=True)
    def prange_test(A):
        s = 0
        for i in prange(A.shape[0]):
            s += A[i]
        return s

Examples
========

In this section, we give an example of how this feature helps
parallelize Logistic Regression::

    @numba.jit(nopython=True, parallel=True)
    def logistic_regression(Y, X, w, iterations):
        for i in range(iterations):
            w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
        return w

We will not discuss details of the algorithm, but instead focus on how
this program behaves with auto-parallelization:

1. Input ``Y`` is a vector of size ``N``, ``X`` is an ``N x D`` matrix,
   and ``w`` is a vector of size ``D``.

2. The function body is an iterative loop that updates variable ``w``.
   The loop body consists of a sequence of vector and matrix operations.

3. The inner ``dot`` operation produces a vector of size ``N``, followed by a
   sequence of arithmetic operations either between a scalar and vector of
   size ``N``, or two vectors both of size ``N``.

4. The outer ``dot`` produces a vector of size ``D``, followed by an inplace
   array subtraction on variable ``w``.

5. With auto-parallelization, all operations that produce array of size
   ``N`` are fused together to become a single parallel kernel. This includes
   the inner ``dot`` operation and all point-wise array operations following it.

6. The outer ``dot`` operation produces a result array of different dimension,
   and is not fused with the above kernel.

Here, the only thing required to take advantage of parallel hardware is to set
the :ref:`parallel_jit_option` option for :func:`~numba.jit`, with no
modifications to the ``logistic_regression`` function itself.  If we were to
give an equivalence parallel implementation using :func:`~numba.guvectorize`,
it would require a pervasive change that rewrites the code to extract kernel
computation that can be parallelized, which was both tedious and challenging.


.. seealso:: :ref:`parallel_jit_option`, :ref:`Parallel FAQs <parallel_FAQs>`
