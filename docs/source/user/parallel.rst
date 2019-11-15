.. Copyright (c) 2017 Intel Corporation
   SPDX-License-Identifier: BSD-2-Clause

.. _numba-parallel:

=======================================
Automatic parallelization with ``@jit``
=======================================

Setting the :ref:`parallel_jit_option` option for :func:`~numba.jit` enables
a Numba transformation pass that attempts to automatically parallelize and
perform other optimizations on (part of) a function. At the moment, this
feature only works on CPUs.

Some operations inside a user defined function, e.g. adding a scalar value to
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

Another feature of the code transformation pass (when ``parallel=True``) is
support for explicit parallel loops. One can use Numba's ``prange`` instead of
``range`` to specify that a loop can be parallelized. The user is required to
make sure that the loop does not have cross iteration dependencies except for
supported reductions.

A reduction is inferred automatically if a variable is updated by a binary
function/operator using its previous value in the loop body. The initial value
of the reduction is inferred automatically for the ``+=``, ``-=``,  ``*=``,
and ``/=`` operators.
For other functions/operators, the reduction variable should hold the identity
value right before entering the ``prange`` loop.  Reductions in this manner
are supported for scalars and for arrays of arbitrary dimensions.

The example below demonstrates a parallel loop with a
reduction (``A`` is a one-dimensional Numpy array)::

    from numba import njit, prange

    @njit(parallel=True)
    def prange_test(A):
        s = 0
        # Without "parallel=True" in the jit-decorator
        # the prange statement is equivalent to range
        for i in prange(A.shape[0]):
            s += A[i]
        return s

The following example demonstrates a product reduction on a two-dimensional array::

    from numba import njit, prange
    import numpy as np

    @njit(parallel=True)
    def two_d_array_reduction_prod(n):
        shp = (13, 17)
        result1 = 2 * np.ones(shp, np.int_)
        tmp = 2 * np.ones_like(result1)

        for i in prange(n):
            result1 *= tmp

        return result1

Care should be taken, however, when reducing into slices or elements of an array 
if the elements specified by the slice or index are written to simultaneously by 
multiple parallel threads. The compiler may not detect such cases and then a race condition
would occur.

The following example demonstrates such a case where a race condition in the execution of the 
parallel for-loop results in an incorrect return value::

    from numba import njit, prange
    import numpy as np

    @njit(parallel=True)
    def prange_wrong_result(x):
        n = x.shape[0]
        y = np.zeros(4)
        for i in prange(n):
            # accumulating into the same element of `y` from different
            # parallel iterations of the loop results in a race condition
            y[:] += x[i]

        return y

as does the following example where the accumulating element is explicitly specified::

    from numba import njit, prange
    import numpy as np

    @njit(parallel=True)
    def prange_wrong_result(x):
        n = x.shape[0]
        y = np.zeros(4)
        for i in prange(n):
            # accumulating into the same element of `y` from different
            # parallel iterations of the loop results in a race condition
            y[i % 4] += x[i]

        return y

whereas performing a whole array reduction is fine::

   from numba import njit, prange
   import numpy as np

   @njit(parallel=True)
   def prange_ok_result_whole_arr(x):
       n = x.shape[0]
       y = np.zeros(4)
       for i in prange(n):
           y += x[i]
       return y

as is creating a slice reference outside of the parallel reduction loop::

   from numba import njit, prange
   import numpy as np

   @njit(parallel=True)
   def prange_ok_result_outer_slice(x):
       n = x.shape[0]
       y = np.zeros(4)
       z = y[:]
       for i in prange(n):
           z += x[i]
       return y

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


.. _numba-parallel-diagnostics:

Diagnostics
===========

.. note:: At present not all parallel transforms and functions can be tracked
          through the code generation process. Occasionally diagnostics about
          some loops or transforms may be missing.

The :ref:`parallel_jit_option` option for :func:`~numba.jit` can produce
diagnostic information about the transforms undertaken in automatically
parallelizing the decorated code. This information can be accessed in two ways,
the first is by setting the environment variable
:envvar:`NUMBA_PARALLEL_DIAGNOSTICS`, the second is by calling
:meth:`~Dispatcher.parallel_diagnostics`, both methods give the same information
and print to ``STDOUT``. The level of verbosity in the diagnostic information is
controlled by an integer argument of value between 1 and 4 inclusive, 1 being
the least verbose and 4 the most. For example::

    @njit(parallel=True)
    def test(x):
        n = x.shape[0]
        a = np.sin(x)
        b = np.cos(a * a)
        acc = 0
        for i in prange(n - 2):
            for j in prange(n - 1):
                acc += b[i] + b[j + 1]
        return acc

    test(np.arange(10))

    test.parallel_diagnostics(level=4)

produces::

    ================================================================================
    ======= Parallel Accelerator Optimizing:  Function test, example.py (4)  =======
    ================================================================================


    Parallel loop listing for  Function test, example.py (4)
    --------------------------------------|loop #ID
    @njit(parallel=True)                  |
    def test(x):                          |
        n = x.shape[0]                    |
        a = np.sin(x)---------------------| #0
        b = np.cos(a * a)-----------------| #1
        acc = 0                           |
        for i in prange(n - 2):-----------| #3
            for j in prange(n - 1):-------| #2
                acc += b[i] + b[j + 1]    |
        return acc                        |
    --------------------------------- Fusing loops ---------------------------------
    Attempting fusion of parallel loops (combines loops with similar properties)...
    Trying to fuse loops #0 and #1:
        - fusion succeeded: parallel for-loop #1 is fused into for-loop #0.
    Trying to fuse loops #0 and #3:
        - fusion failed: loop dimension mismatched in axis 0. slice(0, x_size0.1, 1)
    != slice(0, $40.4, 1)
    ----------------------------- Before Optimization ------------------------------
    Parallel region 0:
    +--0 (parallel)
    +--1 (parallel)


    Parallel region 1:
    +--3 (parallel)
    +--2 (parallel)


    --------------------------------------------------------------------------------
    ------------------------------ After Optimization ------------------------------
    Parallel region 0:
    +--0 (parallel, fused with loop(s): 1)


    Parallel region 1:
    +--3 (parallel)
    +--2 (serial)



    Parallel region 0 (loop #0) had 1 loop(s) fused.

    Parallel region 1 (loop #3) had 0 loop(s) fused and 1 loop(s) serialized as part
    of the larger parallel loop (#3).
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------

    ---------------------------Loop invariant code motion---------------------------

    Instruction hoisting:
    loop #0:
    Failed to hoist the following:
        dependency: $arg_out_var.10 = getitem(value=x, index=$parfor__index_5.99)
        dependency: $0.6.11 = getattr(value=$0.5, attr=sin)
        dependency: $expr_out_var.9 = call $0.6.11($arg_out_var.10, func=$0.6.11, args=[Var($arg_out_var.10, example.py (7))], kws=(), vararg=None)
        dependency: $arg_out_var.17 = $expr_out_var.9 * $expr_out_var.9
        dependency: $0.10.20 = getattr(value=$0.9, attr=cos)
        dependency: $expr_out_var.16 = call $0.10.20($arg_out_var.17, func=$0.10.20, args=[Var($arg_out_var.17, example.py (8))], kws=(), vararg=None)
    loop #3:
    Has the following hoisted:
        $const58.3 = const(int, 1)
        $58.4 = _n_23 - $const58.3
    --------------------------------------------------------------------------------



To aid users unfamiliar with the transforms undertaken when the
:ref:`parallel_jit_option` option is used, and to assist in the understanding of
the subsequent sections, the following definitions are provided:

* Loop fusion
    `Loop fusion <https://en.wikipedia.org/wiki/Loop_fission_and_fusion>`_ is a
    technique whereby loops with equivalent bounds may be combined under certain
    conditions to produce a loop with a larger body (aiming to improve data
    locality).

* Loop serialization
    Loop serialization occurs when any number of ``prange`` driven loops are
    present inside another ``prange`` driven loop. In this case the outermost
    of all the ``prange`` loops executes in parallel and any inner ``prange``
    loops (nested or otherwise) are treated as standard ``range`` based loops.
    Essentially, nested parallelism does not occur.

* Loop invariant code motion
    `Loop invariant code motion
    <https://en.wikipedia.org/wiki/Loop-invariant_code_motion>`_ is an
    optimization technique that analyses a loop to look for statements that can
    be moved outside the loop body without changing the result of executing the
    loop, these statements are then "hoisted" out of the loop to save repeated
    computation.

* Allocation hoisting
    Allocation hoisting is a specialized case of loop invariant code motion that
    is possible due to the design of some common NumPy allocation methods.
    Explanation of this technique is best driven by an example:

    .. code-block:: python

        @njit(parallel=True)
        def test(n):
            for i in prange(n):
                temp = np.zeros((50, 50)) # <--- Allocate a temporary array with np.zeros()
                for j in range(50):
                    temp[j, j] = i

            # ...do something with temp

    internally, this is transformed to approximately the following:

    .. code-block:: python

        @njit(parallel=True)
        def test(n):
            for i in prange(n):
                temp = np.empty((50, 50)) # <--- np.zeros() is rewritten as np.empty()
                temp[:] = 0               # <--- and then a zero initialisation
                for j in range(50):
                    temp[j, j] = i

            # ...do something with temp

    then after hoisting:

    .. code-block:: python

        @njit(parallel=True)
        def test(n):
            temp = np.empty((50, 50)) # <--- allocation is hoisted as a loop invariant as `np.empty` is considered pure
            for i in prange(n):
                temp[:] = 0           # <--- this remains as assignment is a side effect
                for j in range(50):
                    temp[j, j] = i

            # ...do something with temp

    it can be seen that the ``np.zeros`` allocation is split into an allocation
    and an assignment, and then the allocation is hoisted out of the loop in
    ``i``, this producing more efficient code as the allocation only occurs
    once.

The parallel diagnostics report sections
----------------------------------------

The report is split into the following sections:

#. Code annotation
    This is the first section and contains the source code of the decorated
    function with loops that have parallel semantics identified and enumerated.
    The ``loop #ID`` column on the right of the source code lines up with
    identified parallel loops. From the example, ``#0`` is ``np.sin``, ``#1``
    is ``np.cos`` and ``#2`` and ``#3`` are ``prange()``:

    .. code-block:: python

        Parallel loop listing for  Function test, example.py (4)
        --------------------------------------|loop #ID
        @njit(parallel=True)                  |
        def test(x):                          |
            n = x.shape[0]                    |
            a = np.sin(x)---------------------| #0
            b = np.cos(a * a)-----------------| #1
            acc = 0                           |
            for i in prange(n - 2):-----------| #3
                for j in prange(n - 1):-------| #2
                    acc += b[i] + b[j + 1]    |
            return acc                        |

    It is worth noting that the loop IDs are enumerated in the order they are
    discovered which is not necessarily the same order as present in the source.
    Further, it should also be noted that the parallel transforms use a static
    counter for loop ID indexing. As a consequence it is possible for the loop
    ID index to not start at 0 due to use of the same counter for internal
    optimizations/transforms taking place that are invisible to the user.

#. Fusing loops
    This section describes the attempts made at fusing discovered
    loops noting which succeeded and which failed. In the case of failure to
    fuse a reason is given (e.g. dependency on other data). From the example:

    .. code-block:: text

        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Trying to fuse loops #0 and #1:
            - fusion succeeded: parallel for-loop #1 is fused into for-loop #0.
        Trying to fuse loops #0 and #3:
            - fusion failed: loop dimension mismatched in axis 0. slice(0, x_size0.1, 1)
        != slice(0, $40.4, 1)

    It can be seen that fusion of loops ``#0`` and ``#1`` was attempted and this
    succeeded (both are based on the same dimensions of ``x``). Following the
    successful fusion of ``#0`` and ``#1``, fusion was attempted between ``#0``
    (now including the fused ``#1`` loop) and ``#3``. This fusion failed because
    there is a loop dimension mismatch, ``#0`` is size ``x.shape`` whereas
    ``#3`` is size ``x.shape[0] - 2``.

#. Before Optimization
    This section shows the structure of the parallel regions in the code before
    any optimization has taken place, but with loops associated with their final
    parallel region (this is to make before/after optimization output directly
    comparable). Multiple parallel regions may exist if there are loops which
    cannot be fused, in this case code within each region will execute in
    parallel, but each parallel region will run sequentially. From the example:

    .. code-block:: text

        Parallel region 0:
        +--0 (parallel)
        +--1 (parallel)


        Parallel region 1:
        +--3 (parallel)
        +--2 (parallel)

    As alluded to by the `Fusing loops` section, there are necessarily two
    parallel regions in the code. The first contains loops ``#0`` and ``#1``,
    the second contains ``#3`` and ``#2``, all loops are marked ``parallel`` as
    no optimization has taken place yet.

#. After Optimization
    This section shows the structure of the parallel regions in the code after
    optimization has taken place. Again, parallel regions are enumerated with
    their corresponding loops but this time loops which are fused or serialized
    are noted and a summary is presented. From the example:

    .. code-block:: text

        Parallel region 0:
        +--0 (parallel, fused with loop(s): 1)


        Parallel region 1:
        +--3 (parallel)
           +--2 (serial)

        Parallel region 0 (loop #0) had 1 loop(s) fused.

        Parallel region 1 (loop #3) had 0 loop(s) fused and 1 loop(s) serialized as part
        of the larger parallel loop (#3).


    It can be noted that parallel region 0 contains loop ``#0`` and, as seen in
    the `fusing loops` section, loop ``#1`` is fused into loop ``#0``. It can
    also be noted that parallel region 1 contains loop ``#3`` and that loop
    ``#2`` (the inner ``prange()``) has been serialized for execution in the
    body of loop ``#3``.

#. Loop invariant code motion
    This section shows for each loop, after optimization has occurred:

    * the instructions that failed to be hoisted and the reason for failure
      (dependency/impure).
    * the instructions that were hoisted.
    * any allocation hoisting that may have occurred.

    From the example:

    .. code-block:: text

        Instruction hoisting:
        loop #0:
        Failed to hoist the following:
            dependency: $arg_out_var.10 = getitem(value=x, index=$parfor__index_5.99)
            dependency: $0.6.11 = getattr(value=$0.5, attr=sin)
            dependency: $expr_out_var.9 = call $0.6.11($arg_out_var.10, func=$0.6.11, args=[Var($arg_out_var.10, example.py (7))], kws=(), vararg=None)
            dependency: $arg_out_var.17 = $expr_out_var.9 * $expr_out_var.9
            dependency: $0.10.20 = getattr(value=$0.9, attr=cos)
            dependency: $expr_out_var.16 = call $0.10.20($arg_out_var.17, func=$0.10.20, args=[Var($arg_out_var.17, example.py (8))], kws=(), vararg=None)
        loop #3:
        Has the following hoisted:
            $const58.3 = const(int, 1)
            $58.4 = _n_23 - $const58.3

    The first thing to note is that this information is for advanced users as it
    refers to the :term:`Numba IR` of the function being transformed. As an
    example, the expression ``a * a`` in the example source partly translates to
    the expression ``$arg_out_var.17 = $expr_out_var.9 * $expr_out_var.9`` in
    the IR, this clearly cannot be hoisted out of ``loop #0`` because it is not
    loop invariant! Whereas in ``loop #3``, the expression
    ``$const58.3 = const(int, 1)`` comes from the source ``b[j + 1]``, the
    number ``1`` is clearly a constant and so can be hoisted out of the loop.

.. seealso:: :ref:`parallel_jit_option`, :ref:`Parallel FAQs <parallel_FAQs>`
