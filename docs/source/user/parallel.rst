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

A reduction is inferred automatically if a variable is updated by a supported binary
function/operator using its previous value in the loop body.  The following
functions/operators are supported: ``+=``, ``+``, ``-=``, ``-``, ``*=``,
``*``, ``/=``, ``/``, ``max()``, ``min()``.
The initial value of the reduction is inferred automatically for the
supported operators (i.e., not the ``max`` and ``min`` functions).
Note that the ``//=`` operator is not supported because
in the general case the result depends on the order in which the divisors are
applied.  However, if all divisors are integers then the programmer may be
able to rewrite the ``//=`` reduction as a ``*=`` reduction followed by
a single floor division after the parallel region where the divisor is the
accumulated product.
For the ``max`` and ``min`` functions, the reduction variable should hold the identity
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

.. note:: When using Python's ``range`` to induce a loop, Numba types the
          induction variable as a signed integer. This is also the case for
          Numba's ``prange`` when ``parallel=False``. However, for
          ``parallel=True``, if the range is identifiable as strictly positive,
          the type of the induction variable  will be ``uint64``. The impact of
          a ``uint64`` induction variable is often most noticeable when
          undertaking operations involving it and a signed integer. Under
          Numba's type coercion rules, such a case will commonly result in the
          operation producing a floating point result type.

.. note:: Only prange loops with a single entry block and single exit block
          can be converted such that they will be run in parallel.  Exceptional
          control flow, such as an assertion, in the loop can generate multiple
          exit blocks and cause the loop not to be run in parallel.  If this is
          the case, Numba will issue a warning indicating which loop could not
          be parallelized.

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

Unsupported Operations
======================

This section contains a non-exhaustive list of commonly encountered but
currently unsupported features:

#. **Mutating a list is not threadsafe**

   Concurrent write operations on container types (i.e. lists, sets and
   dictionaries) in a ``prange`` parallel region are not threadsafe e.g.::

    @njit(parallel=True)
    def invalid():
        z = []
        for i in prange(10000):
            z.append(i)
        return z

   It is highly likely that the above will result in corruption or an access
   violation as containers require thread-safety under mutation but this feature
   is not implemented.

#. **Induction variables are not associated with thread ID**

   The use of the induction variable induced by a ``prange`` based loop in
   conjunction with ``get_num_threads`` as a method of ensuring safe writes into
   a pre-sized container is not valid e.g.::

    @njit(parallel=True)
    def invalid():
        n = get_num_threads()
        z = [0 for _ in range(n)]
        for i in prange(100):
            z[i % n] += i
        return z

   The above can on occasion appear to work, but it does so by luck. There's no
   guarantee about which indexes are assigned to which executing threads or the
   order in which the loop iterations execute.

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

.. _numba-parallel-scheduling:

Scheduling
==========

By default, Numba divides the iterations of a parallel region into approximately equal
sized chunks and gives one such chunk to each configured thread.
(See :ref:`setting_the_number_of_threads`).
This scheduling approach is equivalent to OpenMP's static schedule with no specified
chunk size and is appropriate when the work required for each iteration is nearly constant.
Conversely, if the work required per iteration, as shown in the ``prange`` loop below,
varies significantly then this static
scheduling approach can lead to load imbalances and longer execution times.

.. literalinclude:: ../../../numba/tests/doc_examples/test_parallel_chunksize.py
   :language: python
   :caption: from ``test_unbalanced_example`` of ``numba/tests/doc_examples/test_parallel_chunksize.py``
   :start-after: magictoken.ex_unbalanced.begin
   :end-before: magictoken.ex_unbalanced.end
   :dedent: 12
   :linenos:

In such cases,
Numba provides a mechanism to control how many iterations of a parallel region
(i.e., the chunk size) go into each chunk.
Numba then computes the number of required chunks which is
equal to the number of iterations divided by the chunk size, truncated to the nearest
integer.  All of these chunks are then approximately, equally sized.
Numba then gives one such chunk to each configured
thread as above and when a thread finishes a chunk, Numba gives that thread the next
available chunk.
This scheduling approach is similar to OpenMP's dynamic scheduling
option with the specified chunk size.
(Note that Numba is only capable of supporting this dynamic scheduling
of parallel regions if the underlying Numba threading backend,
:ref:`numba-threading-layer`, is also capable of dynamic scheduling.
At the moment, only the ``tbb`` backend is capable of dynamic
scheduling and so is required if any performance
benefit is to be achieved from this chunk size selection mechanism.)
To minimize execution time, the programmer must
pick a chunk size that strikes a balance between greater load balancing with smaller
chunk sizes and less scheduling overhead with larger chunk sizes.
See :ref:`chunk-details-label` for additional details on the internal implementation
of chunk sizes.

The number of iterations of a parallel region in a chunk is stored as a thread-local
variable and can be set using
:func:`numba.set_parallel_chunksize`.  This function takes one integer parameter
whose value must be greater than
or equal to 0.  A value of 0 is the default value and instructs Numba to use the
static scheduling approach above.  Values greater than 0 instruct Numba to use that value
as the chunk size in the dynamic scheduling approach described above.
:func:`numba.set_parallel_chunksize` returns the previous value of the chunk size.
The current value of this thread local variable is used as the chunk size for all
subsequent parallel regions invoked by this thread.
However, upon entering a parallel region, Numba sets the chunk size thread local variable
for each of the threads executing that parallel region back to the default of 0,
since it is unlikely
that any nested parallel regions would require the same chunk size.  If the same thread is
used to execute a sequential and parallel region then that thread's chunk size
variable is set to 0 at the beginning of the parallel region and restored to
its original value upon exiting the parallel region.
This behavior is demonstrated in ``func1`` in the example below in that the
reported chunk size inside the ``prange`` parallel region is 0 but is 4 outside
the parallel region.  Note that if the ``prange`` is not executed in parallel for
any reason (e.g., setting ``parallel=False``) then the chunk size reported inside
the non-parallel prange would be reported as 4.
This behavior may initially be counterintuitive to programmers as it differs from
how thread local variables typically behave in other languages.
A programmer may use
the chunk size API described in this section within the threads executing a parallel
region if the programmer wishes to specify a chunk size for any nested parallel regions
that may be launched.
The current value of the parallel chunk size can be obtained by calling
:func:`numba.get_parallel_chunksize`.
Both of these functions can be used from standard Python and from within Numba JIT compiled functions
as shown below.  Both invocations of ``func1`` would be executed with a chunk size of 4 whereas
``func2`` would use a chunk size of 8.

.. literalinclude:: ../../../numba/tests/doc_examples/test_parallel_chunksize.py
   :language: python
   :caption: from ``test_chunksize_manual`` of ``numba/tests/doc_examples/test_parallel_chunksize.py``
   :start-after: magictoken.ex_chunksize_manual.begin
   :end-before: magictoken.ex_chunksize_manual.end
   :dedent: 12
   :linenos:

Since this idiom of saving and restoring is so common, Numba provides the
:func:`parallel_chunksize` with clause context-manager to simplify the idiom.
As shown below, this with clause can be invoked from both standard Python and
within Numba JIT compiled functions.  As with other Numba context-managers, be
aware that the raising of exceptions is not supported from within a context managed
block that is part of a Numba JIT compiled function.

.. literalinclude:: ../../../numba/tests/doc_examples/test_parallel_chunksize.py
   :language: python
   :caption: from ``test_chunksize_with`` of ``numba/tests/doc_examples/test_parallel_chunksize.py``
   :start-after: magictoken.ex_chunksize_with.begin
   :end-before: magictoken.ex_chunksize_with.end
   :dedent: 12
   :linenos:

Note that these functions to set the chunk size only have an effect on
Numba automatic parallelization with the :ref:`parallel_jit_option` option.
Chunk size specification has no effect on the :func:`~numba.vectorize` decorator
or the :func:`~numba.guvectorize` decorator.

.. seealso:: :ref:`parallel_jit_option`, :ref:`Parallel FAQs <parallel_FAQs>`
