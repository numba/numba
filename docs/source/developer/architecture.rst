
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
The control flow graph describes the ways that execution can move from one
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

Examples of attributes that are macro-expanded include the CUDA instrinsics for
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

Stage 6: Rewrite typed IR
-------------------------

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


Stage 7a: Generate nopython LLVM IR
-----------------------------------

If type inference succeeds in finding a Numba type for every intermediate
variable, then Numba can (potentially) generate specialized native code.  This
process is called :term:`lowering`.  The Numba IR tree is translated into
LLVM IR by using helper classes from `llvmlite <http://llvmlite.pydata.org/>`_.
The  machine-generated LLVM IR can seem unnecessarily verbose, but the LLVM
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
