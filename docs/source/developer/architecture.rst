
.. _architecture:

==================
Numba architecture
==================

Introduction
============

Numba is a compiler for Python bytecode with optional type-specialization.

Suppose you enter a function like this into the standard Python interpreter
(henceforward referred to as "CPython")::

    def add(a, b):
        return a + b

The interpreter will immediately parse the function and convert it into a
bytecode representation that describes how the CPython interpreter should
execute the function at a low level.  For the example above, it looks
something like this::

    >>> import dis
    >>> dis.dis(add)
    2           0 LOAD_FAST                0 (a)
                3 LOAD_FAST                1 (b)
                6 BINARY_ADD
                7 RETURN_VALUE


CPython uses a stack-based interpreter (much like an HP calculator), so the
code first pushes two local variables onto the stack.  The ``BINARY_ADD``
opcode pops the top two arguments off the stack and makes a Python C API
function call that is equivalent to calling ``a.__add__(b)``.  The result is
then pushed onto the top of the interpreter stack.  Finally, the
``RETURN_VALUE`` opcode returns value on the top of the stack as the result of
the function call.

Numba can take this bytecode and compile it to machine code that performs the
same operations as the CPython interpreter, treating ``a`` and ``b`` as
generic Python objects.  The full semantics of Python are preserved, and the
compiled function can be used with any kind of objects that have the add
operator defined.  When a Numba function is compiled this way, we say that it
has been compiled in :term:`object mode`, because the code still manipulates
Python objects.

Numba code compiled in object mode is not much faster than executing the
original Python function in the CPython interpreter.  However, if we
specialize the function to only run with certain data types, Numba can
generate much shorter and more efficient code that manipulates the data
natively without any calls into the Python C API.  When code has been compiled
for specific data types so that the function body no longer relies on the
Python runtime, we say the function has been compiled in :term:`nopython mode`.
Numeric code compiled in nopython mode can be hundreds of times faster
than the original Python.


Compiler architecture
=====================

Like many compilers, Numba can be conceptually divided into a
*frontend* and a *backend*.

The Numba *frontend* comprises the stages which analyze the Python bytecode,
translate it to :term:`Numba IR` and perform various transformations and
analysis steps on the IR.  One of the key steps is :term:`type inference`.
The frontend must succeed in typing all variables unambiguously in order
for the backend to generate code in :term:`nopython mode`, because the
backend uses type information to match appropriate code generators with
the values they operate on.

The Numba *backend* walks the Numba IR resulting from the frontend analyses
and exploits the type information deduced by the type inference phase to
produce the right LLVM code for each encountered operation.  After LLVM
code is produced, the LLVM library is asked to optimize it and generate
native processor code for the final, native function.

There are other pieces besides the compiler frontend and backend, such
as the caching machinery for JIT functions.  Those pieces are not considered
in this document.


Contexts
========

Numba is quite flexible, allowing it to generate code for different hardware
architectures like CPUs and GPUs.  In order to support these different
applications, Numba uses a *typing context* and a *target context*.

A *typing context* is used in the compiler frontend to perform type inference
on operations and values in the function.  Similar typing contexts could be
used for many architectures because for nearly all cases, typing inference
is hardware-independent.  However, Numba currently has a different typing
context for each target.

A *target context* is used to generate the specific instruction sequence
required to operate on the Numba types identified during type inference.
Target contexts are architecture-specific and are flexible in defining
the execution model and available Python APIs.  For example, Numba has a "cpu"
and a "cuda" context for those two kinds of architecture, and a "parallel"
context which produces multithreaded CPU code.


Compiler stages
===============

The :func:`~numba.jit` decorator in Numba ultimately calls
``numba.compiler.compile_extra()`` which compiles the Python function in a
multi-stage process, described below.

Stage 1: Analyze bytecode
-------------------------

At the start of compilation, the function bytecode is passed to an instance of
the Numba interpreter (``numba.interpreter``).  The interpreter object
analyzes the bytecode to find the control flow graph (``numba.controlflow``).
The control flow graph (CFG) describes the ways that execution can move from one
block to the next inside the function as a result of loops and branches.

The data flow analysis (``numba.dataflow``) takes the control flow graph and
traces how values get pushed and popped off the Python interpreter stack for
different code paths.  This is important to understand the lifetimes of
variables on the stack, which are needed in Stage 2.

If you set the environment variable ``NUMBA_DUMP_CFG`` to 1, Numba will dump
the results of the control flow graph analysis to the screen.  Our ``add()``
example is pretty boring, since there is only one statement block::

    CFG adjacency lists:
    {0: []}
    CFG dominators:
    {0: set([0])}
    CFG post-dominators:
    {0: set([0])}
    CFG back edges: []
    CFG loops:
    {}
    CFG node-to-loops:
    {0: []}

A function with more complex flow control will have a more interesting
control flow graph.  This function::

    def doloops(n):
        acc = 0
        for i in range(n):
            acc += 1
            if n == 10:
                break
        return acc

compiles to this bytecode::

      9           0 LOAD_CONST               1 (0)
                  3 STORE_FAST               1 (acc)

     10           6 SETUP_LOOP              46 (to 55)
                  9 LOAD_GLOBAL              0 (range)
                 12 LOAD_FAST                0 (n)
                 15 CALL_FUNCTION            1
                 18 GET_ITER
            >>   19 FOR_ITER                32 (to 54)
                 22 STORE_FAST               2 (i)

     11          25 LOAD_FAST                1 (acc)
                 28 LOAD_CONST               2 (1)
                 31 INPLACE_ADD
                 32 STORE_FAST               1 (acc)

     12          35 LOAD_FAST                0 (n)
                 38 LOAD_CONST               3 (10)
                 41 COMPARE_OP               2 (==)
                 44 POP_JUMP_IF_FALSE       19

     13          47 BREAK_LOOP
                 48 JUMP_ABSOLUTE           19
                 51 JUMP_ABSOLUTE           19
            >>   54 POP_BLOCK

     14     >>   55 LOAD_FAST                1 (acc)
                 58 RETURN_VALUE

The corresponding CFG for this bytecode is::

    CFG adjacency lists:
    {0: [6], 6: [19], 19: [54, 22], 22: [19, 47], 47: [55], 54: [55], 55: []}
    CFG dominators:
    {0: set([0]),
     6: set([0, 6]),
     19: set([0, 6, 19]),
     22: set([0, 6, 19, 22]),
     47: set([0, 6, 19, 22, 47]),
     54: set([0, 6, 19, 54]),
     55: set([0, 6, 19, 55])}
    CFG post-dominators:
    {0: set([0, 6, 19, 55]),
     6: set([6, 19, 55]),
     19: set([19, 55]),
     22: set([22, 55]),
     47: set([47, 55]),
     54: set([54, 55]),
     55: set([55])}
    CFG back edges: [(22, 19)]
    CFG loops:
    {19: Loop(entries=set([6]), exits=set([54, 47]), header=19, body=set([19, 22]))}
    CFG node-to-loops:
    {0: [], 6: [], 19: [19], 22: [19], 47: [], 54: [], 55: []}

The numbers in the CFG refer to the bytecode offsets shown just to the left
of the opcode names above.

.. _arch_generate_numba_ir:

Stage 2: Generate the Numba IR
------------------------------

Once the control flow and data analyses are complete, the Numba interpreter
can step through the bytecode and translate it into an Numba-internal
intermediate representation.  This translation process changes the function
from a stack machine representation (used by the Python interpreter) to a
register machine representation (used by LLVM).

Although the IR is stored in memory as a tree of objects, it can be serialized
to a string for debugging.  If you set the environment variable
``NUMBA_DUMP_IR`` equal to 1, the Numba IR will be dumped to the screen.  For
the ``add()`` function described above, the Numba IR looks like::

   label 0:
       a = arg(0, name=a)                       ['a']
       b = arg(1, name=b)                       ['b']
       $0.3 = a + b                             ['$0.3', 'a', 'b']
       del b                                    []
       del a                                    []
       $0.4 = cast(value=$0.3)                  ['$0.3', '$0.4']
       del $0.3                                 []
       return $0.4                              ['$0.4']

The ``del`` instructions are produced by :ref:`live variable analysis`.
Those instructions ensure references are not leaked.
In :term:`nopython mode`, some objects are tracked by the numba runtime and
some are not.  For tracked objects, a dereference operation is emitted;
otherwise, the instruction is an no-op.
In :term:`object mode` each variable contains an owned reference to a PyObject.


Stage 3: Macro expansion
------------------------

Now that the function has been translated into the Numba IR, macro expansion can
be performed. Macro expansion converts specific attributes that are known to
Numba into IR nodes representing function calls. This is initiated in the
``numba.compiler.translate_stage`` function, and is implemented in
``numba.macro``.

Examples of attributes that are macro-expanded include the CUDA intrinsics for
grid, block and thread dimensions and indices. For example, the assignment to
``tx`` in the following function::

  @cuda.jit(argtypes=[f4[:]])
  def f(a):
      tx = cuda.threadIdx.x

has the following representation after translation to Numba IR::

  $0.1 = global(cuda: <module 'numba.cuda' from '...'>) ['$0.1']
  $0.2 = getattr(value=$0.1, attr=threadIdx) ['$0.1', '$0.2']
  del $0.1                                 []
  $0.3 = getattr(value=$0.2, attr=x)       ['$0.2', '$0.3']
  del $0.2                                 []
  tx = $0.3                                ['$0.3', 'tx']

After macro expansion, the ``$0.3 = getattr(value=$0.2, attr=x)`` IR node is
translated into::

  $0.3 = call tid.x(, )                    ['$0.3']

which represents an instance of the ``Intrinsic`` IR node for calling the
``tid.x`` intrinsic function.

.. _`rewrite-untyped-ir`:

Stage 4: Rewrite untyped IR
---------------------------

Before running type inference, it may be desired to run certain
transformations on the Numba IR.  One such example is to detect ``raise``
statements which have an implicitly constant argument, so as to
support them in :term:`nopython mode`.  Let's say you compile the
following function with Numba::

   def f(x):
      if x == 0:
         raise ValueError("x cannot be zero")

If you set the :envvar:`NUMBA_DUMP_IR` environment variable to ``1``,
you'll see the IR being rewritten before the type inference phase::

   REWRITING:
       del $0.3                                 []
       $12.1 = global(ValueError: <class 'ValueError'>) ['$12.1']
       $const12.2 = const(str, x cannot be zero) ['$const12.2']
       $12.3 = call $12.1($const12.2)           ['$12.1', '$12.3', '$const12.2']
       del $const12.2                           []
       del $12.1                                []
       raise $12.3                              ['$12.3']
   ____________________________________________________________
       del $0.3                                 []
       $12.1 = global(ValueError: <class 'ValueError'>) ['$12.1']
       $const12.2 = const(str, x cannot be zero) ['$const12.2']
       $12.3 = call $12.1($const12.2)           ['$12.1', '$12.3', '$const12.2']
       del $const12.2                           []
       del $12.1                                []
       raise <class 'ValueError'>('x cannot be zero') []


.. _arch_type_inference:

Stage 5: Infer types
--------------------

Now that the Numba IR has been generated and macro-expanded, type analysis
can be performed.  The types of the function arguments can be taken either
from the explicit function signature given in the ``@jit`` decorator
(such as ``@jit('float64(float64, float64)')``), or they can be taken from
the types of the actual function arguments if compilation is happening
when the function is first called.

The type inference engine is found in ``numba.typeinfer``.  Its job is to
assign a type to every intermediate variable in the Numba IR.  The result of
this pass can be seen by setting the :envvar:`NUMBA_DUMP_ANNOTATION`
environment variable to 1:

.. code-block:: python

   -----------------------------------ANNOTATION-----------------------------------
   # File: archex.py
   # --- LINE 4 ---

   @jit(nopython=True)

   # --- LINE 5 ---

   def add(a, b):

       # --- LINE 6 ---
       # label 0
       #   a = arg(0, name=a)  :: int64
       #   b = arg(1, name=b)  :: int64
       #   $0.3 = a + b  :: int64
       #   del b
       #   del a
       #   $0.4 = cast(value=$0.3)  :: int64
       #   del $0.3
       #   return $0.4

       return a + b


If type inference fails to find a consistent type assignment for all the
intermediate variables, it will label every variable as type ``pyobject`` and
fall back to object mode.  Type inference can fail when unsupported Python
types, language features, or functions are used in the function body.


.. _`rewrite-typed-ir`:

Stage 6a: Rewrite typed IR
--------------------------

This pass's purpose is to perform any high-level optimizations that still
require, or could at least benefit from, Numba IR type information.

One example of a problem domain that isn't as easily optimized once
lowered is the domain of multidimensional array operations.  When
Numba lowers an array operation, Numba treats the operation like a
full ufunc kernel.  During lowering a single array operation, Numba
generates an inline broadcasting loop that creates a new result array.
Then Numba generates an application loop that applies the operator
over the array inputs.  Recognizing and rewriting these loops once
they are lowered into LLVM is hard, if not impossible.

An example pair of optimizations in the domain of array operators is
loop fusion and shortcut deforestation.  When the optimizer
recognizes that the output of one array operator is being fed into
another array operator, and only to that array operator, it can fuse
the two loops into a single loop.  The optimizer can further eliminate
the temporary array allocated for the initial operation by directly
feeding the result of the first operation into the second, skipping
the store and load to the intermediate array.  This elimination is
known as shortcut deforestation.  Numba currently uses the rewrite
pass to implement these array optimizations.  For more information,
please consult the ":ref:`case-study-array-expressions`" subsection,
later in this document.

One can see the result of rewriting by setting the
:envvar:`NUMBA_DUMP_IR` environment variable to a non-zero value (such
as 1).  The following example shows the output of the rewrite pass as
it recognizes an array expression consisting of a multiply and add,
and outputs a fused kernel as a special operator, :func:`arrayexpr`::

  ______________________________________________________________________
  REWRITING:
  a0 = arg(0, name=a0)                     ['a0']
  a1 = arg(1, name=a1)                     ['a1']
  a2 = arg(2, name=a2)                     ['a2']
  $0.3 = a0 * a1                           ['$0.3', 'a0', 'a1']
  del a1                                   []
  del a0                                   []
  $0.5 = $0.3 + a2                         ['$0.3', '$0.5', 'a2']
  del a2                                   []
  del $0.3                                 []
  $0.6 = cast(value=$0.5)                  ['$0.5', '$0.6']
  del $0.5                                 []
  return $0.6                              ['$0.6']
  ____________________________________________________________
  a0 = arg(0, name=a0)                     ['a0']
  a1 = arg(1, name=a1)                     ['a1']
  a2 = arg(2, name=a2)                     ['a2']
  $0.5 = arrayexpr(ty=array(float64, 1d, C), expr=('+', [('*', [Var(a0, test.py (14)), Var(a1, test.py (14))]), Var(a2, test.py (14))])) ['$0.5', 'a0', 'a1', 'a2']
  del a0                                   []
  del a1                                   []
  del a2                                   []
  $0.6 = cast(value=$0.5)                  ['$0.5', '$0.6']
  del $0.5                                 []
  return $0.6                              ['$0.6']
  ______________________________________________________________________

Following this rewrite, Numba lowers the array expression into a new
ufunc-like function that is inlined into a single loop that only
allocates a single result array.


.. _`parallel-accelerator`:

Stage 6b: Perform Automatic Parallelization
-------------------------------------------

This pass is only performed if the ``parallel`` option in the :func:`~numba.jit`
decorator is set to ``True``.  This pass find parallelism implicit in the
semantics of operations in the Numba IR and replaces those operations
with explicitly parallel representations of those operations using a
special `parfor` operator.  Then, optimizations are performed to maximize
the number of parfors that are adjacent to each other such that they can
then be fused together into one parfor that takes only one pass over the
data and will thus typically have better cache performance.  Finally,
during lowering, these parfor operators are converted to a form similar
to guvectorize to implement the actual parallelism.

The automatic parallelization pass has a number of sub-passes, many of
which are controllable using a dictionary of options passed via the
``parallel`` keyword argument to :func:`~numba.jit`::

   { 'comprehension': True/False,  # parallel comprehension
     'prange':        True/False,  # parallel for-loop
     'numpy':         True/False,  # parallel numpy calls
     'reduction':     True/False,  # parallel reduce calls
     'setitem':       True/False,  # parallel setitem
     'stencil':       True/False,  # parallel stencils
     'fusion':        True/False,  # enable fusion or not
   }

The default is set to `True` for all of them. The sub-passes are
described in more detail in the following paragraphs.

#. CFG Simplification
    Sometimes Numba IR will contain chains of blocks containing no loops which
    are merged in this sub-pass into single blocks.  This sub-pass simplifies
    subsequent analysis of the IR.

#. Numpy canonicalization
    Some Numpy operations can be written as operations on Numpy objects (e.g.
    ``arr.sum()``), or as calls to Numpy taking those objects (e.g.
    ``numpy.sum(arr)``).  This sub-pass converts all such operations to the
    latter form for cleaner subsequent analysis.

#. Array analysis
    A critical requirement for later parfor fusion is that parfors have
    identical iteration spaces and these iteration spaces typically correspond
    to the sizes of the dimensions of Numpy arrays.  In this sub-pass, the IR is
    analyzed to determine equivalence classes for the dimensions of Numpy
    arrays.  Consider the example, ``a = b + 1``, where ``a`` and ``b`` are both
    Numpy arrays.  Here, we know that each dimension of ``a`` must have the same
    equivalence class as the corresponding dimension of ``b``.  Typically,
    routines rich in Numpy operations will enable equivalence classes to be
    fully known for all arrays created within a function.

    Array analysis will also reason about size equivalvence for slice selection,
    and boolean array masking (one dimensional only). For example, it is able to
    infer that ``a[1 : n-1]`` is of the same size as ``b[0 : n-2]``.

    Array analysis may also insert safety assumptions to ensure pre-conditions
    related to array sizes are met before an operation can be parallelized.
    For example, ``np.dot(X, w)`` between a 2-D matrix ``X`` and a 1-D vector ``w``
    requires that the second dimension of ``X`` is of the same size as ``w``.
    Usually this kind of runtime check is automatically inserted, but if array
    analysis can infer such equivalence, it will skip them.

    Users can even help array analysis by turning implicit knowledge about
    array sizes into explicit assertions. For example, in the code below:

    .. code-block:: python

       @numba.njit(parallel=True)
       def logistic_regression(Y, X, w, iterations):
           assert(X.shape == (Y.shape[0], w.shape[0]))
           for i in range(iterations):
               w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)
           return w

    Making the explicit assertion helps eliminate all bounds checks in the
    rest of the function.

#. ``prange()`` to parfor
    The use of prange (:ref:`numba-prange`) in a for loop is an explicit
    indication from the programmer that all iterations of the for loop can
    execute in parallel.  In this sub-pass, we analyze the CFG to locate loops
    and to convert those loops controlled by a prange object to the explicit
    `parfor` operator.  Each explicit parfor operator consists of:

    a. A list of loop nest information that describes the iteration space of the
       parfor.  Each entry in the loop nest list contains an indexing variable,
       the start of the range, the end of the range, and the step value for each
       iteration.
    #. An initialization (init) block which contains instructions to be executed
       one time before the parfor begins executing.
    #. A loop body comprising a set of basic blocks that correspond to the body
       of the loop and compute one point in the iteration space.
    #. The index variables used for each dimension of the iteration space.

    For parfor `pranges`, the loop nest is a single entry where the start,
    stop, and step fields come from the specified `prange`.  The init block is
    empty for `prange` parfors and the loop body is the set of blocks in the
    loop minus the loop header.

    With parallelization on, array comprehensions (:ref:`pysupported-comprehension`)
    will also be translated to prange so as to run in parallel. This behavior
    be disabled by setting ``parallel={'comprehension': False}``.

    Likewise, the overall `prange` to `parfor` translation can be disabled by
    setting ``parallel={'prange': False}``, in which case `prange` is treated the
    same as `range`.

#. Numpy to parfor
    In this sub-pass, Numpy functions such as ``ones``, ``zeros``, ``dot``, most
    of the random number generating functions, arrayexprs (from Section
    :ref:`rewrite-typed-ir`), and Numpy reductions are converted to parfors.
    Generally, this conversion creates the loop nest list, whose length is equal
    to the number of dimensions of the left-hand side of the assignment
    instruction in the IR.  The number and size of the dimensions of the
    left-hand-side array is taken from the array analysis information generated
    in sub-pass 3 above.  An instruction to create the result Numpy array is
    generated and stored in the new parfor's init block.  A basic block is
    created for the loop body and an instruction is generated and added to the
    end of that block to store the result of the computation into the array at
    the current point in the iteration space.  The result stored into the array
    depends on the operation that is being converted.  For example, for ``ones``,
    the value stored is a constant 1.  For calls to generate a random array, the
    value comes from a call to the same random number function but with the size
    parameter dropped and therefore returning a scalar.  For arrayexpr operators,
    the arrayexpr tree is converted to Numba IR and the value at the root of that
    expression tree is used to write into the output array. The translation from
    Numpy functions and arrayexpr operators to `parfor` can be disabled by
    setting ``parallel={'numpy': False}``.

    For reductions, the loop nest list is similarly created using the array
    analysis information for the array being reduced.  In the init block, the
    initial value is assigned to the reduction variable.  The loop body consists
    of a single block in which the next value in the iteration space is fetched
    and the reduction operation is applied to that value and the current
    reduction value and the result stored back into the reduction value.
    The translation of reduction functions to `parfor` can be disabled by
    setting ``parallel={'reduction': False}``.

    Setting the :envvar:`NUMBA_DEBUG_ARRAY_OPT_STATS` environment variable to
    1 will show some statistics about parfor conversions in general.

#. Setitem to parfor
    Setting a range of array elements using a slice or boolean array selection
    can also run in parallel.  Statement such as ``A[P] = B[Q]``
    (or a simpler case ``A[P] = c``, where ``c`` is a scalar) is translated to
    `parfor` if one of the following conditions is met:

     a. ``P`` and ``Q`` are slices or multi-dimensional selector involving
        scalar and slices, and ``A[P]`` and ``B[Q]`` are considered size
        equivalent by array analysis. Only 2-value slice/range is supported,
        3-value with a step will not be translated to `parfor`.
     #. ``P`` and ``Q`` are the same boolean array.

    This translation can be disabled by setting ``parallel={'setitem': False}``.

#. Simplification
    Performs a copy propagation and dead code elimination pass.

#. Fusion
    This sub-pass first processes each basic block and does a reordering of the
    instructions within the block with the goal of pushing parfors lower in the
    block and lifting non-parfors towards the start of the block.  In practice,
    this approach does a good job of getting parfors adjacent to each other in
    the IR, which enables more parfors to then be fused.  During parfor fusion,
    each basic block is repeatedly scanned until no further fusion is possible.
    During this scan, each set of adjacent instructions are considered.
    Adjacent instructions are fused together if:

    a. they are both parfors
    #. the parfors' loop nests are the same size and the array equivalence
       classes for each dimension of the loop nests are the same, and
    #. the first parfor does not create a reduction variable used by the
       second parfor.

    The two parfors are fused together by adding the second parfor's init block
    to the first's, merging the two parfors' loop bodies together and replacing
    the instances of the second parfor's loop index variables in the second
    parfor's body with the loop index variables for the first parfor.
    Fusion can be disabled by setting ``parallel={'fusion': False}``.

    Setting the :envvar:`NUMBA_DEBUG_ARRAY_OPT_STATS` environment variable to
    1 will show some statistics about parfor fusions.

#. Push call objects and compute parfor parameters
    In the lowering phase described in Section :ref:`lowering`, each parfor
    becomes a separate function executed in parallel in ``guvectorize``
    (:ref:`guvectorize`) style.  Since parfors may use variables defined
    previously in a function, when those parfors become separate functions,
    those variables must be passed to the parfor function as parameters.  In
    this sub-pass, a use-def scan is made over each parfor body and liveness
    information is used to determine which variables are used but not defined by
    the parfor.  That list of variables is stored here in the parfor for use
    during lowering.  Function variables are a special case in this process
    since function variables cannot be passed to functions compiled in nopython
    mode.  Instead, for function variables, this sub-pass pushes the assignment
    instruction to the function variable into the parfor body so that those do
    not need to be passed as parameters.

    To see the intermediate IR between the above sub-passes and other debugging
    information, set the :envvar:`NUMBA_DEBUG_ARRAY_OPT` environment variable to
    1. For the example in Section :ref:`rewrite-typed-ir`, the following IR with
    a parfor is generated during this stage::

     ______________________________________________________________________
     label 0:
         a0 = arg(0, name=a0)                     ['a0']
         a0_sh_attr0.0 = getattr(attr=shape, value=a0) ['a0', 'a0_sh_attr0.0']
         $consta00.1 = const(int, 0)              ['$consta00.1']
         a0size0.2 = static_getitem(value=a0_sh_attr0.0, index_var=$consta00.1, index=0) ['$consta00.1', 'a0_sh_attr0.0', 'a0size0.2']
         a1 = arg(1, name=a1)                     ['a1']
         a1_sh_attr0.3 = getattr(attr=shape, value=a1) ['a1', 'a1_sh_attr0.3']
         $consta10.4 = const(int, 0)              ['$consta10.4']
         a1size0.5 = static_getitem(value=a1_sh_attr0.3, index_var=$consta10.4, index=0) ['$consta10.4', 'a1_sh_attr0.3', 'a1size0.5']
         a2 = arg(2, name=a2)                     ['a2']
         a2_sh_attr0.6 = getattr(attr=shape, value=a2) ['a2', 'a2_sh_attr0.6']
         $consta20.7 = const(int, 0)              ['$consta20.7']
         a2size0.8 = static_getitem(value=a2_sh_attr0.6, index_var=$consta20.7, index=0) ['$consta20.7', 'a2_sh_attr0.6', 'a2size0.8']
     ---begin parfor 0---
     index_var =  parfor_index.9
     LoopNest(index_variable=parfor_index.9, range=0,a0size0.2,1 correlation=5)
     init block:
         $np_g_var.10 = global(np: <module 'numpy' from '/usr/local/lib/python3.5/dist-packages/numpy/__init__.py'>) ['$np_g_var.10']
         $empty_attr_attr.11 = getattr(attr=empty, value=$np_g_var.10) ['$empty_attr_attr.11', '$np_g_var.10']
         $np_typ_var.12 = getattr(attr=float64, value=$np_g_var.10) ['$np_g_var.10', '$np_typ_var.12']
         $0.5 = call $empty_attr_attr.11(a0size0.2, $np_typ_var.12, kws=(), func=$empty_attr_attr.11, vararg=None, args=[Var(a0size0.2, test2.py (7)), Var($np_typ_var.12, test2.py (7))]) ['$0.5', '$empty_attr_attr.11', '$np_typ_var.12', 'a0size0.2']
     label 1:
         $arg_out_var.15 = getitem(value=a0, index=parfor_index.9) ['$arg_out_var.15', 'a0', 'parfor_index.9']
         $arg_out_var.16 = getitem(value=a1, index=parfor_index.9) ['$arg_out_var.16', 'a1', 'parfor_index.9']
         $arg_out_var.14 = $arg_out_var.15 * $arg_out_var.16 ['$arg_out_var.14', '$arg_out_var.15', '$arg_out_var.16']
         $arg_out_var.17 = getitem(value=a2, index=parfor_index.9) ['$arg_out_var.17', 'a2', 'parfor_index.9']
         $expr_out_var.13 = $arg_out_var.14 + $arg_out_var.17 ['$arg_out_var.14', '$arg_out_var.17', '$expr_out_var.13']
         $0.5[parfor_index.9] = $expr_out_var.13  ['$0.5', '$expr_out_var.13', 'parfor_index.9']
     ----end parfor 0----
         $0.6 = cast(value=$0.5)                  ['$0.5', '$0.6']
         return $0.6                              ['$0.6']
     ______________________________________________________________________

  .. _`lowering`:

Stage 7a: Generate nopython LLVM IR
-----------------------------------

If type inference succeeds in finding a Numba type for every intermediate
variable, then Numba can (potentially) generate specialized native code.  This
process is called :term:`lowering`.  The Numba IR tree is translated into
LLVM IR by using helper classes from `llvmlite <http://llvmlite.pydata.org/>`_.
The machine-generated LLVM IR can seem unnecessarily verbose, but the LLVM
toolchain is able to optimize it quite easily into compact, efficient code.

The basic lowering algorithm is generic, but the specifics of how particular
Numba IR nodes are translated to LLVM instructions is handled by the
target context selected for compilation.  The default target context is
the "cpu" context, defined in ``numba.targets.cpu``.

The LLVM IR can be displayed by setting the :envvar:`NUMBA_DUMP_LLVM` environment
variable to 1.  For the "cpu" context, our ``add()`` example would look like:

.. code-block:: llvm

   define i32 @"__main__.add$1.int64.int64"(i64* %"retptr",
                                            {i8*, i32}** %"excinfo",
                                            i8* %"env",
                                            i64 %"arg.a", i64 %"arg.b")
   {
      entry:
        %"a" = alloca i64
        %"b" = alloca i64
        %"$0.3" = alloca i64
        %"$0.4" = alloca i64
        br label %"B0"
      B0:
        store i64 %"arg.a", i64* %"a"
        store i64 %"arg.b", i64* %"b"
        %".8" = load i64* %"a"
        %".9" = load i64* %"b"
        %".10" = add i64 %".8", %".9"
        store i64 %".10", i64* %"$0.3"
        %".12" = load i64* %"$0.3"
        store i64 %".12", i64* %"$0.4"
        %".14" = load i64* %"$0.4"
        store i64 %".14", i64* %"retptr"
        ret i32 0
   }

The post-optimization LLVM IR can be output by setting
:envvar:`NUMBA_DUMP_OPTIMIZED` to 1.  The optimizer shortens the code
generated above quite significantly:

.. code-block:: llvm

   define i32 @"__main__.add$1.int64.int64"(i64* nocapture %retptr,
                                            { i8*, i32 }** nocapture readnone %excinfo,
                                            i8* nocapture readnone %env,
                                            i64 %arg.a, i64 %arg.b)
   {
      entry:
        %.10 = add i64 %arg.b, %arg.a
        store i64 %.10, i64* %retptr, align 8
        ret i32 0
   }

If created during :ref:`parallel-accelerator`, parfor operations are
lowered in the following manner.  First, instructions in the parfor's init
block are lowered into the existing function using the normal lowering code.
Second, the loop body of the parfor is turned into a separate GUFunc.
Third, code is emitted for the current function to call the parallel GUFunc.

To create a GUFunc from the parfor body, the signature of the GUFunc is
created by taking the parfor parameters as identified in step 9 of
Stage :ref:`parallel-accelerator` and adding to that a special `schedule`
parameter, across which the GUFunc will be parallelized.  The schedule
parameter is in effect a static schedule mapping portions of the parfor
iteration space to Numba threads and so the length of the schedule
array is the same as the number of configured Numba threads.  To make
this process easier and somewhat less dependent on changes to Numba IR,
this stage creates a Python function as text that contains the parameters
to the GUFunc and iteration code that takes the current schedule entry
and loops through the specified portion of the iteration space.  In the
body of that loop, a special sentinel is inserted for subsequent easy
location.  This code that handles the processing of the iteration space
is then ``eval``'ed into existence and the Numba compiler's run_frontend
function is called to generate IR.  That IR is scanned to locate the
sentinel and the sentinel is replaced with the loop body of the parfor.
Then, the process of creating the parallel GUFunc is completed by
compiling this merged IR with the Numba compiler's ``compile_ir`` function.

To call the parallel GUFunc, the static schedule must be created.
Code is inserted to call a function named ``do_scheduling.``  This function
is called with the size of each of the parfor's dimensions and the number
`N` of configured Numba threads (:envvar:`NUMBA_NUM_THREADS`).
The ``do_scheduling`` function will divide
the iteration space into N approximately equal sized regions (linear for
1D, rectangular for 2D, or hyperrectangles for 3+D) and the resulting
schedule is passed to the parallel GUFunc.  The number of threads
dedicated to a given dimension of the full iteration space is roughly
proportional to the ratio of the size of the given dimension to the sum
of the sizes of all the dimensions of the iteration space.

Parallel reductions are not natively provided by GUFuncs but the parfor
lowering strategy allows us to use GUFuncs in a way that reductions can
be performed in parallel.  To accomplish this, for each reduction variable
computed by a parfor, the parallel GUFunc and the code that calls it are
modified to make the scalar reduction variable into an array of reduction
variables whose length is equal to the number of Numba threads.  In addition,
the GUFunc still contains a scalar version of the reduction variable that
is updated by the parfor body during each iteration.  One time at the
end of the GUFunc this local reduction variable is copied into the
reduction array.  In this way, false sharing of the reduction array is
prevented.  Code is also inserted into the main
function after the parallel GUFunc has returned that does a reduction
across this smaller reduction array and this final reduction value is
then stored into the original scalar reduction variable.

The GUFunc corresponding to the example from Section :ref:`parallel-accelerator`
can be seen below::

  ______________________________________________________________________
  label 0:
      sched.29 = arg(0, name=sched)            ['sched.29']
      a0 = arg(1, name=a0)                     ['a0']
      a1 = arg(2, name=a1)                     ['a1']
      a2 = arg(3, name=a2)                     ['a2']
      _0_5 = arg(4, name=_0_5)                 ['_0_5']
      $3.1.24 = global(range: <class 'range'>) ['$3.1.24']
      $const3.3.21 = const(int, 0)             ['$const3.3.21']
      $3.4.23 = getitem(value=sched.29, index=$const3.3.21) ['$3.4.23', '$const3.3.21', 'sched.29']
      $const3.6.28 = const(int, 1)             ['$const3.6.28']
      $3.7.27 = getitem(value=sched.29, index=$const3.6.28) ['$3.7.27', '$const3.6.28', 'sched.29']
      $const3.8.32 = const(int, 1)             ['$const3.8.32']
      $3.9.31 = $3.7.27 + $const3.8.32         ['$3.7.27', '$3.9.31', '$const3.8.32']
      $3.10.36 = call $3.1.24($3.4.23, $3.9.31, kws=[], func=$3.1.24, vararg=None, args=[Var($3.4.23, <string> (2)), Var($3.9.31, <string> (2))]) ['$3.1.24', '$3.10.36', '$3.4.23', '$3.9.31']
      $3.11.30 = getiter(value=$3.10.36)       ['$3.10.36', '$3.11.30']
      jump 1                                   []
  label 1:
      $28.2.35 = iternext(value=$3.11.30)      ['$28.2.35', '$3.11.30']
      $28.3.25 = pair_first(value=$28.2.35)    ['$28.2.35', '$28.3.25']
      $28.4.40 = pair_second(value=$28.2.35)   ['$28.2.35', '$28.4.40']
      branch $28.4.40, 2, 3                    ['$28.4.40']
  label 2:
      $arg_out_var.15 = getitem(value=a0, index=$28.3.25) ['$28.3.25', '$arg_out_var.15', 'a0']
      $arg_out_var.16 = getitem(value=a1, index=$28.3.25) ['$28.3.25', '$arg_out_var.16', 'a1']
      $arg_out_var.14 = $arg_out_var.15 * $arg_out_var.16 ['$arg_out_var.14', '$arg_out_var.15', '$arg_out_var.16']
      $arg_out_var.17 = getitem(value=a2, index=$28.3.25) ['$28.3.25', '$arg_out_var.17', 'a2']
      $expr_out_var.13 = $arg_out_var.14 + $arg_out_var.17 ['$arg_out_var.14', '$arg_out_var.17', '$expr_out_var.13']
      _0_5[$28.3.25] = $expr_out_var.13        ['$28.3.25', '$expr_out_var.13', '_0_5']
      jump 1                                   []
  label 3:
      $const44.1.33 = const(NoneType, None)    ['$const44.1.33']
      $44.2.39 = cast(value=$const44.1.33)     ['$44.2.39', '$const44.1.33']
      return $44.2.39                          ['$44.2.39']
  ______________________________________________________________________


Stage 7b: Generate object mode LLVM IR
--------------------------------------

If type inference fails to find Numba types for all values inside a function,
the function will be compiled in object mode.  The generated LLVM will be
significantly longer, as the compiled code will need to make calls to the
`Python C API <https://docs.python.org/3/c-api/>`_ to perform basically all
operations.  The optimized LLVM for our example ``add()`` function is:

.. code-block:: llvm

   @PyExc_SystemError = external global i8
   @".const.Numba_internal_error:_object_mode_function_called_without_an_environment" = internal constant [73 x i8] c"Numba internal error: object mode function called without an environment\00"
   @".const.name_'a'_is_not_defined" = internal constant [24 x i8] c"name 'a' is not defined\00"
   @PyExc_NameError = external global i8
   @".const.name_'b'_is_not_defined" = internal constant [24 x i8] c"name 'b' is not defined\00"

   define i32 @"__main__.add$1.pyobject.pyobject"(i8** nocapture %retptr, { i8*, i32 }** nocapture readnone %excinfo, i8* readnone %env, i8* %arg.a, i8* %arg.b) {
   entry:
     %.6 = icmp eq i8* %env, null
     br i1 %.6, label %entry.if, label %entry.endif, !prof !0

   entry.if:                                         ; preds = %entry
     tail call void @PyErr_SetString(i8* @PyExc_SystemError, i8* getelementptr inbounds ([73 x i8]* @".const.Numba_internal_error:_object_mode_function_called_without_an_environment", i64 0, i64 0))
     ret i32 -1

   entry.endif:                                      ; preds = %entry
     tail call void @Py_IncRef(i8* %arg.a)
     tail call void @Py_IncRef(i8* %arg.b)
     %.21 = icmp eq i8* %arg.a, null
     br i1 %.21, label %B0.if, label %B0.endif, !prof !0

   B0.if:                                            ; preds = %entry.endif
     tail call void @PyErr_SetString(i8* @PyExc_NameError, i8* getelementptr inbounds ([24 x i8]* @".const.name_'a'_is_not_defined", i64 0, i64 0))
     tail call void @Py_DecRef(i8* null)
     tail call void @Py_DecRef(i8* %arg.b)
     ret i32 -1

   B0.endif:                                         ; preds = %entry.endif
     %.30 = icmp eq i8* %arg.b, null
     br i1 %.30, label %B0.endif1, label %B0.endif1.1, !prof !0

   B0.endif1:                                        ; preds = %B0.endif
     tail call void @PyErr_SetString(i8* @PyExc_NameError, i8* getelementptr inbounds ([24 x i8]* @".const.name_'b'_is_not_defined", i64 0, i64 0))
     tail call void @Py_DecRef(i8* %arg.a)
     tail call void @Py_DecRef(i8* null)
     ret i32 -1

   B0.endif1.1:                                      ; preds = %B0.endif
     %.38 = tail call i8* @PyNumber_Add(i8* %arg.a, i8* %arg.b)
     %.39 = icmp eq i8* %.38, null
     br i1 %.39, label %B0.endif1.1.if, label %B0.endif1.1.endif, !prof !0

   B0.endif1.1.if:                                   ; preds = %B0.endif1.1
     tail call void @Py_DecRef(i8* %arg.a)
     tail call void @Py_DecRef(i8* %arg.b)
     ret i32 -1

   B0.endif1.1.endif:                                ; preds = %B0.endif1.1
     tail call void @Py_DecRef(i8* %arg.b)
     tail call void @Py_DecRef(i8* %arg.a)
     tail call void @Py_IncRef(i8* %.38)
     tail call void @Py_DecRef(i8* %.38)
     store i8* %.38, i8** %retptr, align 8
     ret i32 0
   }

   declare void @PyErr_SetString(i8*, i8*)

   declare void @Py_IncRef(i8*)

   declare void @Py_DecRef(i8*)

   declare i8* @PyNumber_Add(i8*, i8*)


The careful reader might notice several unnecessary calls to ``Py_IncRef``
and ``Py_DecRef`` in the generated code.  Currently Numba isn't able to
optimize those away.

Object mode compilation will also attempt to identify loops which can be
extracted and statically-typed for "nopython" compilation.  This process is
called *loop-lifting*, and results in the creation of a hidden nopython mode
function just containing the loop which is then called from the original
function.  Loop-lifting helps improve the performance of functions that
need to access uncompilable code (such as I/O or plotting code) but still
contain a time-intensive section of compilable code.

Stage 8: Compile LLVM IR to machine code
----------------------------------------

In both :term:`object mode` and :term:`nopython mode`, the generated LLVM IR
is compiled by the LLVM JIT compiler and the machine code is loaded into
memory.  A Python wrapper is also created (defined in
``numba.dispatcher.Dispatcher``) which can do the dynamic dispatch to the
correct version of the compiled function if multiple type specializations
were generated (for example, for both ``float32`` and ``float64`` versions
of the same function).

The machine assembly code generated by LLVM can be dumped to the screen by
setting the :envvar:`NUMBA_DUMP_ASSEMBLY` environment variable to 1:

.. code-block:: gas

           .globl  __main__.add$1.int64.int64
           .align  16, 0x90
           .type   __main__.add$1.int64.int64,@function
   __main__.add$1.int64.int64:
           addq    %r8, %rcx
           movq    %rcx, (%rdi)
           xorl    %eax, %eax
           retq

The assembly output will also include the generated wrapper function that
translates the Python arguments to native data types.
