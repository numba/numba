
.. _arch-generators:

===================
Notes on generators
===================

Numba recently gained support for compiling generator functions.  This
document explains some of the implementation choices.


Terminology
===========

For clarity, we distinguish between *generator functions* and
*generators*.  A generator function is a function containing one or
several ``yield`` statements.  A generator (sometimes also called "generator
iterator") is the return value of a generator function; it resumes
execution inside its frame each time :py:func:`next` is called.

A *yield point* is the place where a ``yield`` statement is called.
A *resumption point* is the place just after a *yield point* where execution
is resumed when :py:func:`next` is called again.


Function analysis
=================

Suppose we have the following simple generator function::

   def gen(x, y):
       yield x + y
       yield x - y

Here is its CPython bytecode, as printed out using :py:func:`dis.dis`::

  7           0 LOAD_FAST                0 (x)
              3 LOAD_FAST                1 (y)
              6 BINARY_ADD
              7 YIELD_VALUE
              8 POP_TOP

  8           9 LOAD_FAST                0 (x)
             12 LOAD_FAST                1 (y)
             15 BINARY_SUBTRACT
             16 YIELD_VALUE
             17 POP_TOP
             18 LOAD_CONST               0 (None)
             21 RETURN_VALUE

When compiling this function with :envvar:`NUMBA_DUMP_IR` set to 1, the
following information is printed out::

   ----------------------------------IR DUMP: gen----------------------------------
   label 0:
       x = arg(0, name=x)                       ['x']
       y = arg(1, name=y)                       ['y']
       $0.3 = x + y                             ['$0.3', 'x', 'y']
       $0.4 = yield $0.3                        ['$0.3', '$0.4']
       del $0.4                                 []
       del $0.3                                 []
       $0.7 = x - y                             ['$0.7', 'x', 'y']
       del y                                    []
       del x                                    []
       $0.8 = yield $0.7                        ['$0.7', '$0.8']
       del $0.8                                 []
       del $0.7                                 []
       $const0.9 = const(NoneType, None)        ['$const0.9']
       $0.10 = cast(value=$const0.9)            ['$0.10', '$const0.9']
       del $const0.9                            []
       return $0.10                             ['$0.10']
   ------------------------------GENERATOR INFO: gen-------------------------------
   generator state variables: ['$0.3', '$0.7', 'x', 'y']
   yield point #1: live variables = ['x', 'y'], weak live variables = ['$0.3']
   yield point #2: live variables = [], weak live variables = ['$0.7']


What does it mean? The first part is the Numba IR, as already seen in
:ref:`arch_generate_numba_ir`.  We can see the two yield points (``yield $0.3``
and ``yield $0.7``).

The second part shows generator-specific information.  To understand it
we have to understand what suspending and resuming a generator means.

When suspending a generator, we are not merely returning a value to the
caller (the operand of the ``yield`` statement).  We also have to save the
generator's *current state* in order to resume execution.  In trivial use
cases, perhaps the CPU's register values or stack slots would be preserved
until the next call to next().  However, any non-trivial case will hopelessly
clobber those values, so we have to save them in a well-defined place.

What are the values we need to save?  Well, in the context of the Numba
Intermediate Representation, we must save all *live variables* at each
yield point.  These live variables are computed thanks to the control
flow graph.

Once live variables are saved and the generator is suspended, resuming
the generator simply involves the inverse operation: the live variables
are restored from the saved generator state.

.. note::
   It is the same analysis which helps insert Numba ``del`` instructions
   where appropriate.

Let's go over the generator info again::

   generator state variables: ['$0.3', '$0.7', 'x', 'y']
   yield point #1: live variables = ['x', 'y'], weak live variables = ['$0.3']
   yield point #2: live variables = [], weak live variables = ['$0.7']

Numba has computed the union of all live variables (denoted as "state
variables").  This will help define the layout of the :ref:`generator
structure <generator-structure>`.  Also, for each yield point, we have
computed two sets of variables:

* the *live variables* are the variables which are used by code following
  the resumption point (i.e. after the ``yield`` statement)

* the *weak live variables* are variables which are del'ed immediately
  after the resumption point; they have to be saved in :term:`object mode`,
  to ensure proper reference cleanup


.. _generator-structure:

The generator structure
=======================

Layout
------

Function analysis helps us gather enough information to define the
layout of the generator structure, which will store the entire execution
state of a generator.  Here is a sketch of the generator structure's layout,
in pseudo-code::

   struct gen_struct_t {
      int32_t resume_index;
      struct gen_args_t {
         arg_0_t arg0;
         arg_1_t arg1;
         ...
         arg_N_t argN;
      }
      struct gen_state_t {
         state_0_t state_var0;
         state_1_t state_var1;
         ...
         state_N_t state_varN;
      }
   }

Let's describe those fields in order.

* The first member, the *resume index*, is an integer telling the generator
  at which resumption point execution must resume.  By convention, it can
  have two special values: 0 means execution must start at the beginning of
  the generator (i.e. the first time :py:func:`next` is called); -1 means
  the generator is exhauted and resumption must immediately raise StopIteration.
  Other values indicate the yield point's index starting from 1
  (corresponding to the indices shown in the generator info above).

* The second member, the *arguments structure* is read-only after it is first
  initialized.  It stores the values of the arguments the generator function
  was called with.  In our example, these are the values of ``x`` and ``y``.

* The third member, the *state structure*, stores the live variables as
  computed above.

Concretely, our example's generator structure (assuming the generator
function is called with floating-point numbers) is then::

   struct gen_struct_t {
      int32_t resume_index;
      struct gen_args_t {
         double arg0;
         double arg1;
      }
      struct gen_state_t {
         double $0.3;
         double $0.7;
         double x;
         double y;
      }
   }

Note that here, saving ``x`` and ``y`` is redundant: Numba isn't able to
recognize that the state variables ``x`` and ``y`` have the same value
as ``arg0`` and ``arg1``.

Allocation
----------

How does Numba ensure the generator structure is preserved long enough?
There are two cases:

* When a Numba-compiled generator function is called from a Numba-compiled
  function, the structure is allocated on the stack by the callee.  In this
  case, generator instantiation is practically costless.

* When a Numba-compiled generator function is called from regular Python
  code, a CPython-compatible wrapper is instantiated that has the right
  amount of allocated space to store the structure, and whose
  :c:member:`~PyTypeObject.tp_iternext` slot is a wrapper around the
  generator's native code.


Compiling to native code
========================

When compiling a generator function, three native functions are actually
generated by Numba:

* An initialization function.  This is the function corresponding
  to the generator function itself: it receives the function arguments and
  stores them inside the generator structure (which is passed by pointer).
  It also initialized the *resume index* to 0, indicating that the generator
  hasn't started yet.

* A next() function.  This is the function called to resume execution
  inside the generator.  Its single argument is a pointer to the generator
  structure and it returns the next yielded value (or a special exit code
  is used if the generator is exhausted, for quick checking when called
  from Numba-compiled functions).

* An optional finalizer.  In object mode, this function ensures that all
  live variables stored in the generator state are decref'ed, even if the
  generator is destroyed without having been exhausted.

The next() function
-------------------

The next() function is the least straight-forward of the three native
functions.  It starts with a trampoline which dispatches execution to the
right resume point depending on the *resume index* stored in the generator
structure.  Here is how the function start may look like in our example:

.. code-block:: llvm

   define i32 @"__main__.gen.next"(
      double* nocapture %retptr,
      { i8*, i32 }** nocapture readnone %excinfo,
      i8* nocapture readnone %env,
      { i32, { double, double }, { double, double, double, double } }* nocapture %arg.gen)
   {
     entry:
        %gen.resume_index = getelementptr { i32, { double, double }, { double, double, double, double } }* %arg.gen, i64 0, i32 0
        %.47 = load i32* %gen.resume_index, align 4
        switch i32 %.47, label %stop_iteration [
          i32 0, label %B0
          i32 1, label %generator_resume1
          i32 2, label %generator_resume2
        ]

     ; rest of the function snipped

(uninteresting stuff trimmed from the LLVM IR to make it more readable)

We recognize the pointer to the generator structure in ``%arg.gen``.
The trampoline switch has three targets (one for each *resume index* 0, 1
and 2), and a fallback target label named ``stop_iteration``.  Label ``B0``
represents the function's start, ``generator_resume1`` (resp.
``generator_resume2``) is the resumption point after the first
(resp. second) yield point.

After generation by LLVM, the whole native assembler code for this function
may look like this (on x86-64):

.. code-block:: asm

           .globl  __main__.gen.next
           .align  16, 0x90
   __main__.gen.next:
           movl    (%rcx), %eax
           cmpl    $2, %eax
           je      .LBB1_5
           cmpl    $1, %eax
           jne     .LBB1_2
           movsd   40(%rcx), %xmm0
           subsd   48(%rcx), %xmm0
           movl    $2, (%rcx)
           movsd   %xmm0, (%rdi)
           xorl    %eax, %eax
           retq
   .LBB1_5:
           movl    $-1, (%rcx)
           jmp     .LBB1_6
   .LBB1_2:
           testl   %eax, %eax
           jne     .LBB1_6
           movsd   8(%rcx), %xmm0
           movsd   16(%rcx), %xmm1
           movaps  %xmm0, %xmm2
           addsd   %xmm1, %xmm2
           movsd   %xmm1, 48(%rcx)
           movsd   %xmm0, 40(%rcx)
           movl    $1, (%rcx)
           movsd   %xmm2, (%rdi)
           xorl    %eax, %eax
           retq
   .LBB1_6:
           movl    $-3, %eax
           retq

Note the function returns 0 to indicate a value is yielded, -3 to indicate
StopIteration. ``%rcx`` points to the start of the generator structure,
where the resume index is stored.
