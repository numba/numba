==================
Numba Architecture
==================

Introduction
============

Numba is a compiler for Python bytecode with optional type-specialization.

Suppose you type a function like this into the standard Python interpreter
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
compiled function can used with any kind of objects that have the add operator
defined.  When a Numba function is compiled this way, we say that it has been
compiled in :term:`object mode`, because the code still manipulates Python
objects.

Numba code compiled in object mode is not much faster than executing the
original Python function in the CPython interpreter.  However, if we
specialize the function to only run with certain data types, Numba can
generate much shorter and more efficient code that manipulates the data
natively without any calls into the Python C API.  When code has been compiled
for specific data types so that the function body no longer relies on the
Python runtime, we say the function has been compiled in :term:`nopython mode`.
Numeric code compiled in nopython mode can be hundreds of times faster
than the original Python. 

Contexts
========

Numba is quite flexible, allowing it to generate code for different hardware
architectures like CPUs and GPUs (just CUDA, for now).  In order to support
these different applications, Numba uses a *typing context* and a *target
context*.

A typing context is used in the compiler frontend to perform type inference on
values in the function.  Similar typing contexts could be used for many
architectures because for nearly all cases, typing inference is hardware-independent.
However, Numba currently has a different typing context for each target.

A target context is used to generate the specific instruction sequence
required to operate on the Numba types identified during type inference.
Target contexts are architecture specific.  For example, Numba has a "cpu" and
a "gpu" context, and NumbaPro adds a "parallel" context which produces
multithreaded CPU code.

Compiler Stages
===============

The ``@jit`` decorator in Numba ultimately calls
``numba.compiler.compile_extra()`` which compiles the Python function in a
multi-stage process, described below.

Stage 1: Analyze Bytecode
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
        a.1 = a
        b.1 = b
        $0.3 = a.1 + b.1
        return $0.3

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

Stage 4: Infer Types
--------------------

Now that the Numba IR has been generated and macro-expanded, type analysis
can be performed.  The types of the function arguments can be taken either
from the explicit function signature given in the ``@jit`` decorator 
(such as ``@jit('float64(float64, float64)')``), or they can be taken from
the types of the actual function arguments if compilation is happening
when the function is first called.

The type inference engine is found in ``numba.typeinfer``.  Its job is to
assign a type to every intermediate variable in the Numba IR.  The result of
this pass can be seen by setting the ``NUMBA_DUMP_ANNOTATION`` environment
variable to 1::

    -----------------------------------ANNOTATION-----------------------------------
    # File: test.py
    # --- LINE 3 ---

    @numba.jit()

    # --- LINE 4 ---

    def add(a, b):

        # --- LINE 5 ---
        # label 0
        #   a.1 = a  :: int64
        #   b.1 = b  :: int64
        #   $0.3 = a.1 + b.1  :: int64
        #   return $0.3

        return a + b


    ================================================================================

If type inference fails to find a consistent type assignment for all the
intermediate variables, it will label every variable as type ``pyobject`` and
fall back to object mode.  Type inference can fail when unsupported Python
types, language features, or functions are used in the function body.


Stage 5a: Generate No-Python LLVM IR
------------------------------------

If type inference succeeds in finding a Numba type for every intermediate
variable, then Numba can (potentially) generate specialized native code.  This
process is called *lowering*.  The Numba IR tree is translated into LLVM IR by
using helper classes from `llvmpy <http://www.llvmpy.org/>`_.  The  machine-
generated LLVM IR can seem unnecessarily verbose, but the LLVM  toolchain is
able to optimize it quite easily into compact, efficient code.

The basic lowering algorithm is generic, but the specifics of how particular
Numba IR nodes are translated to LLVM instructions is handled by the
target context selected for compilation.  The default target context is
the "cpu" context, defined in ``numba.targets.cpu``.

The LLVM IR can be displayed by setting the ``NUMBA_DUMP_LLVM`` environment
variable to 1.  For the "cpu" context, our ``add()`` example would look like:

.. code-block:: llvm

    ; ModuleID = 'module.add$3'

    define i32 @add.int64.int64(i64*, i8* %env, i64 %arg.a, i64 %arg.b) {
    entry:
      %a = alloca i64
      store i64 %arg.a, i64* %a
      %b = alloca i64
      store i64 %arg.b, i64* %b
      %a.1 = alloca i64
      %b.1 = alloca i64
      %"$0.3" = alloca i64
      br label %B0

    B0:                                               ; preds = %entry
      %1 = load i64* %a
      store i64 %1, i64* %a.1
      %2 = load i64* %b
      store i64 %2, i64* %b.1
      %3 = load i64* %a.1
      %4 = load i64* %b.1
      %5 = add i64 %3, %4
      store i64 %5, i64* %"$0.3"
      %6 = load i64* %"$0.3"
      store i64 %6, i64* %0
      ret i32 0
    }

The post-optimization LLVM IR can be output by setting ``NUMBA_DUMP_FUNC_OPT``
to 1.  The optimizer shortens the code generated above quite significantly:

.. code-block:: llvm

    ; ModuleID = 'module.add$3'

    define i32 @add.int64.int64(i64*, i8* %env, i64 %arg.a, i64 %arg.b) {
    entry:
      %1 = add i64 %arg.a, %arg.b
      store i64 %1, i64* %0
      ret i32 0
    }

Stage 5b: Generate Object Mode LLVM IR
--------------------------------------

If type inference fails to find Numba types for all values inside a function,
the function will be compiled in object mode.  The generated LLVM will be
significantly longer, as the compiled code will need to make calls to the
`Python C API <https://docs.python.org/3/c-api/>`_ to perform basically all
operations.  The optimized LLVM for our example ``add()`` function is:

.. code-block:: llvm

    ; ModuleID = 'module.add$3'

    @PyExc_SystemError = external global i8
    @".const.Numba internal error: object mode function called without an environment" = internal constant [73 x i8] c"Numba internal error: object mode function called without an environment\00"

    define i32 @add.pyobject.pyobject(i8**, i8* %env, i8* %arg.a, i8* %arg.b) {
    entry:
      call void @Py_IncRef(i8* %arg.a)
      call void @Py_DecRef(i8* null)
      call void @Py_IncRef(i8* %arg.b)
      call void @Py_DecRef(i8* null)
      %1 = icmp eq i8* null, %env
      br i1 %1, label %entry.if, label %entry.endif, !prof !0

    error:                                            ; preds = %entry.endif, %entry.if
      %a.1.0 = phi i8* [ null, %entry.if ], [ %arg.a, %entry.endif ]
      %b.1.0 = phi i8* [ null, %entry.if ], [ %arg.b, %entry.endif ]
      call void @Py_DecRef(i8* %arg.a)
      call void @Py_DecRef(i8* null)
      call void @Py_DecRef(i8* %b.1.0)
      call void @Py_DecRef(i8* %arg.b)
      call void @Py_DecRef(i8* %a.1.0)
      ret i32 -1

    entry.if:                                         ; preds = %entry
      call void @PyErr_SetString(i8* @PyExc_SystemError, i8* getelementptr inbounds ([73 x i8]* @".const.Numba internal error: object mode function called without an environment", i32 0, i32 0))
      br label %error

    entry.endif:                                      ; preds = %entry
      %2 = ptrtoint i8* %env to i64
      %3 = add i64 %2, 16
      %4 = inttoptr i64 %3 to i8*
      call void @Py_IncRef(i8* %arg.a)
      call void @Py_DecRef(i8* null)
      call void @Py_IncRef(i8* %arg.b)
      call void @Py_DecRef(i8* null)
      %5 = call i8* @PyNumber_Add(i8* %arg.a, i8* %arg.b)
      %6 = icmp eq i8* null, %5
      br i1 %6, label %error, label %B0.endif, !prof !0

    B0.endif:                                         ; preds = %entry.endif
      call void @Py_DecRef(i8* null)
      call void @Py_IncRef(i8* %5)
      call void @Py_DecRef(i8* %arg.a)
      call void @Py_DecRef(i8* %5)
      call void @Py_DecRef(i8* %arg.b)
      call void @Py_DecRef(i8* %arg.b)
      call void @Py_DecRef(i8* %arg.a)
      store i8* %5, i8** %0
      ret i32 0
    }

    declare void @Py_IncRef(i8*)

    declare void @Py_DecRef(i8*)

    declare void @PyErr_SetString(i8*, i8*)

    declare i8* @PyNumber_Add(i8*, i8*)

    !0 = metadata !{metadata !"branch_weights", i32 1, i32 99}

The careful reader might notice a lot of unnecessary calls to ``Py_IncRef``
and ``Py_DecRef`` in the generated code.  A special pass is run after the 
LLVM optimizer to identify and remove these extra reference count calls.

Object mode compilation will also attempt to identify loops which can be
extracted and statically-typed for "nopython" compilation.  This process is
called *loop-lifting*, and results in the creation of a hidden nopython mode
function just containing the loop which is then called from the original
function.  Loop-lifting helps improve the performance of functions that
need to access uncompilable code (such as I/O or plotting code) but still
contain a time-intensive section of compilable code.

Stage 6: Compile LLVM IR to Machine Code
----------------------------------------

In both "object mode" and "nopython mode", the generated LLVM IR is compiled
by the LLVM JIT compiler and the machine code is loaded into memory.  A Python
wrapper is also created (defined in ``numba.dispatcher.Overloaded``) which can
do the dynamic dispatch to the correct version of the compiled function if
multiple type specializations were generated (for example, for both
``float32`` and ``float64`` versions of the same function).

The machine assembly code generated by LLVM can be dumped to the screen by
setting the ``NUMBA_DUMP_ASSEMBLY`` environment variable to 1:

.. code-block:: gas

      .section  __TEXT,__text,regular,pure_instructions
      .globl  _add.int64.int64
      .align  4, 0x90
    _add.int64.int64:
      addq  %rcx, %rdx
      movq  %rdx, (%rdi)
      xorl  %eax, %eax
      ret


The assembly output will also include the generated wrapper function that
translates the Python arguments to native data types.
