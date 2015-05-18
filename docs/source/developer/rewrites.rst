=====================================================
Using the Numba Rewrite Pass for Fun and Optimization
=====================================================

Overview
========

This section introduces intermediate representation (IR) rewrites, and
how they can be used to implement optimizations.

As discussed earlier in ":ref:`rewrite-typed-ir`", rewriting the Numba
IR allows us to perform optimizations that would be much more
difficult to perform at the lower LLVM level.  Similar to the Numba
type and lowering subsystems, the rewrite subsystem is user
extensible.  This extensibility affords Numba the possibility of
supporting a wide variety of domain-specific optimizations (DSO's).

The remaining subsections detail the mechanics of implementing a
rewrite, registering a rewrite with the rewrite registry, and provide
examples of adding new rewrites, as well as internals of the array
expression optimization pass.  We conclude by reviewing some use cases
exposed in the examples, as well as reviewing any points where
developers should take care.


Rewriting Passes
================

Rewriting passes have a simple :func:`~Rewrite.match` and
:func:`~Rewrite.apply` interface.  The division between matching and
rewriting follows how one would define a term rewrite in a declarative
domain-specific languages (DSL's).  In such DSL's, one may write a
rewrite as follows::

  <match> => <replacement>


The ``<match>`` and ``<replacement>`` symbols represent IR term
expressions, where the left-hand side presents a pattern to match, and
the right-hand side an IR term constructor to build upon matching.
Whenever the rewrite matches an IR pattern, any free variables in the
left-hand side are bound within a custom environment.  When applied,
the rewrite uses the pattern matching environment to bind any free
variables in the right-hand side.

As Python is not commonly used in a declarative capacity, Numba uses
object state to handle the transfer of information between the
matching and application steps.


The :class:`Rewrite` Base Class
-------------------------------

.. class:: Rewrite

   The :class:`Rewrite` class simply defines an abstract base class
   for Numba rewrites.  Developers should define rewrites as
   subclasses of this base type, overloading the
   :func:`~Rewrite.match` and :func:`~Rewrite.apply` methods.

   .. attribute:: pipeline

       The pipeline attribute contains the
       :class:`numba.compiler.Pipeline` instance that is currently
       compiling the function under consideration for rewriting.

   .. method:: __init__(self, pipeline, *args, **kws)

       The base constructor for rewrites simply stashes its arguments
       into attributes of the same name.  Unless being used in
       debugging or testing, rewrites should only be constructed by
       the :class:`RewriteRegistry` in the
       :func:`RewriteRegistry.apply` method, and the construction
       interface should remain stable (though the pipeline will
       commonly contain just about everything there is to know).

   .. method:: match(self, block, typemap, callmap)

      The :func:`~Rewrite.match` method takes three arguments other
      than *self*:

      * *block*: This is an instance of :class:`numba.ir.Block`.  The
        matching method should iterate over the instructions contained
        in the :attr:`numba.ir.Block.body` member.

      * *typemap*: This is a Python :class:`dict` instance mapping
        from symbol names in the IR, represented as strings, to Numba
        types.

      * *callmap*: This is another :class:`dict` instance mapping from
        calls, represented as :class:`numba.ir.Expr` instances, to
        their corresponding call site type signatures, represented as
        a :class:`numba.typing.templates.Signature` instance.

      The :func:`~Rewrite.match` method should return a :class:`bool`
      result.  A :obj:`True` result should indicate that one or more
      matches were found, and the :func:`~Rewrite.apply` method will
      return a new replacement :class:`numba.ir.Block` instance.  A
      :obj:`False` result should indicate that no matches were found, and
      subsequent calls to :func:`~Rewrite.apply` will return undefined
      or invalid results.

   .. method:: apply(self)

      The :func:`~Rewrite.apply` method should only be invoked
      following a successful call to :func:`~Rewrite.match`.  This
      method takes no additional parameters other than *self*, and
      should return a replacement :class:`numba.ir.Block` instance.

      As mentioned above, the behavior of calling
      :func:`~Rewrite.apply` is undefined unless
      :func:`~Rewrite.match` has already been called and returned
      :obj:`True`.


Subclassing :class:`Rewrite`
----------------------------

Before going into the expectations for the overloaded methods any
:class:`Rewrite` subclass must have, let's step back a minute to
review what is taking place here.  By providing an extensible
compiler, Numba opens itself to user-defined code generators which may
be incomplete, or worse, incorrect.  When a code generator goes awry,
it can cause abnormal program behavior or early termination.
User-defined rewrites add a new level of complexity because they must
not only generate correct code, but the code they generate should
ensure that the compiler does not get stuck in a match/apply loop.
Non-termination by the compiler will directly lead to non-termination
of user function calls.

There are several ways to help ensure that a rewrite terminates:

* *Typing*: A rewrite should generally attempt to decompose composite
  types, and avoid composing new types.  If the rewrite is matching a
  specific type, changing expression types to a lower-level type will
  ensure they will no long match after the rewrite is applied.

* *Special instructions*: A rewrite may synthesize custom operators or
  use special functions in the target IR.  This technique again
  generates code that is no longer within the domain of the original
  match, and the rewrite will terminate.

In the ":ref:`case-study-array-expressions`" subsection, below, we'll
see how the array expression rewriter uses both of these techniques.


Overloading :func:`Rewrite.match`
---------------------------------

Every rewrite developer should seek to have their implementation of
:func:`~Rewrite.match` return a :obj:`False` value as quickly as
possible.  Numba is a just-in-time compiler, and adding compilation
time ultimately adds to the user's run time.  When a rewrite returns
:obj:`False` for a given block, the registry will no longer process that
block with that rewrite, and the compiler is that much closer to
proceeding to lowering.

This need for timeliness has to be balanced against collecting the
necessary information to make a match for a rewrite.  Rewrite
developers should be comfortable adding dynamic attributes to their
subclasses, and then having these new attributes guide construction of
the replacement basic block.


Overloading :func:`Rewrite.apply`
-----------------------------------

The :func:`~Rewrite.apply` method should return a replacement
:class:`numba.ir.Block` instance to replace the basic block that
contained a match for the rewrite.  As mentioned above, the IR built
by :func:`~Rewrite.apply` methods should preserve the semantics of the
user's code, but also seek to avoid generating another match for the
same rewrite or set of rewrites.


The Rewrite Registry
====================

When you want to include a rewrite in the rewrite pass, you should
register it with the rewrite registry.  The :mod:`numba.rewrites`
module provides both the abstract base class and a class decorator for
hooking into the Numba rewrite subsystem.  The following illustrates a
stub definition of a new rewrite::

  from numba import rewrites

  @rewrites.register_rewrite
  class MyRewrite(rewrites.Rewrite):

      def match(self, block, typemap, calltypes):
          raise NotImplementedError("FIXME")

      def apply(self):
          raise NotImplementedError("FIXME")


Developers should note that using the class decorator as shown above
will register a rewrite at import time.  It is the developer's
responsibility to ensure their extensions are loaded before
compilation starts.


.. _`case-study-array-expressions`:

Case study: Array Expressions
=============================

This subsection looks at the array expression rewriter in more depth.
The array expression rewriter, and most of its support functionality,
are found in the :mod:`numba.npyufunc.array_exprs` module.  The
rewriting pass itself is implemented in the :class:`RewriteArrayExprs`
class.  In addition to the rewriter, the
:mod:`~numba.npyufunc.array_exprs` module includes a function for
lowering array expressions,
:func:`~numba.npyufunc.array_exprs._lower_array_expr`.  The overall
optimization process is as follows:

* :func:`RewriteArrayExprs.match`: The rewrite pass looks for two or
  more array operations that form an array expression.

* :func:`RewriteArrayExprs.apply`: Once an array expression is found,
  the rewriter replaces the individual array operations with a new
  kind of IR expression, the ``arrayexpr``.

* :func:`numba.npyufunc.array_exprs._lower_array_expr`: During
  lowering, the code generator calls
  :func:`~numba.npyufunc.array_exprs._lower_array_expr` whenever it
  finds an ``arrayexpr`` IR expression.

More details on each step of the optimization are given below.


The :func:`RewriteArrayExprs.match` method
------------------------------------------

The array expression optimization pass starts by looking for array
operations, including calls to supported :class:`~numpy.ufunc`\'s and
user-defined :class:`~numba.DUFunc`\'s.  Numba IR follows the
conventions of a static single assignment (SSA) language, meaning that
the search for array operators begins with looking for assignment
instructions.

When the rewriting pass calls the :func:`RewriteArrayExprs.match`
method, it first checks to see if it can trivially reject the basic
block.  If the method determines the block to be a candidate for
matching, it sets up the following state variables in the rewrite
object:

* *crnt_block*: The current basic block being matched.

* *typemap*: The *typemap* for the function being matched.

* *matches*: A list of variable names that reference array expressions.

* *array_assigns*: A map from assignment variable names to the actual
  assignment instructions that define the given variable.

* *const_assigns*: A map from assignment variable names to the
  constant valued expression that defines the constant variable.

At this point, the match method iterates iterates over the assignment
instructions in the input basic block.  For each assignment
instruction, the matcher looks for one of two things:

* Array operations: If the right-hand side of the assignment
  instruction is an expression, and the result of that expression is
  an array type, the matcher checks to see if the expression is either
  a known array operation, or a call to a universal function.  If an
  array operator is found, the matcher stores the left-hand variable
  name and the whole instruction in the *array_assigns* member.
  Finally, the matcher tests to see if any operands of the array
  operation have also been identified as targets of other array
  operations.  If one or more operands are also targets of array
  operations, then the matcher will also append the left-hand side
  variable name to the *matches* member.

* Constants: Constants (even scalars) can be operands to array
  operations.  Without worrying about the constant being apart of an
  array expression, the matcher stores constant names and values in
  the *const_assigns* member.

The end of the matching method simply checks for a non-empty *matches*
list, returning :obj:`True` if there were one or more matches, and
:obj:`False` when *matches* is empty.


The :func:`RewriteArrayExprs.apply` method
------------------------------------------

When one or matching array expressions are found by
:func:`RewriteArrayExprs.match`, the rewriting pass will call
:func:`RewriteArrayExprs.apply`.  The apply method works in two
passes.  The first pass iterates over the matches found, and builds a
map from instructions in the old basic block to new instructions in
the new basic block.  The second pass iterates over the instructions
in the old basic block, copying instructions that are not changed by
the rewrite, and replacing or deleting instructions that were
identified by the first pass.

The :func:`RewriteArrayExprs._handle_matches` implements the first
pass of the code generation portion of the rewrite.  For each match,
this method builds a special IR expression that contains an expression
tree for the array expression.  To compute the leaves of the
expression tree, the :func:`~RewriteArrayExprs._handle_matches` method
iterates over the operands of the identified root operation.  If the
operand is another array operation, it is translated into an
expression sub-tree.  If the operand is a constant,
:func:`~RewriteArrayExprs._handle_matches` copies the constant value.
Otherwise, the operand is marked as being used by an array expression.
As the method builds array expression nodes, it builds a map from old
instructions to new instructions (*replace_map*), as well as sets of
variables that may have moved (*used_vars*), and variables that should
be removed altogether (*dead_vars*).  These three data structures are
returned back to the calling :func:`RewriteArrayExprs.apply` method.

The remaining part of the :func:`RewriteArrayExprs.apply` method
iterates over the instructions in the old basic block.  For each
instruction, this method either replaces, deletes, or duplicates that
instruction based on the results of
:func:`RewriteArrayExprs._handle_matches`.  The following list
describes how the optimization handles individual instructions:

* When an instruction is an assignment,
  :func:`~RewriteArrayExprs.apply` checks to see if it is in the
  replacement instruction map.  When an assignment instruction is found
  in the instruction map, :func:`~RewriteArrayExprs.apply` must then
  check to see if the replacement instruction is also in the replacement
  map.  The optimizer continues this check until it either arrives at a
  :obj:`None` value or an instruction that isn't in the replacement map.
  Instructions that have a replacement that is :obj:`None` are deleted.
  Instructions that have a non-:obj:`None` replacement are replaced.
  Assignment instructions not in the replacement map are appended to the
  new basic block with no changes made.

* When the instruction is a delete instruction, the rewrite checks to
  see if it deletes a variable that may still be used by a later array
  expression, or if it deletes a dead variable.  Delete instructions for
  used variables are added to a map of deferred delete instructions that
  :func:`~RewriteArrayExprs.apply` uses to move them past any uses of
  that variable.  The loop copies delete instructions for non-dead
  variables, and ignores delete instructions for dead variables
  (effectively removing them from the basic block).

* All other instructions are appended to the new basic block.

Finally, the :func:`~RewriteArrayExprs.apply` method returns the new
basic block for lowering.


The :func:`~numba.npyufunc.array_exprs._lower_array_expr` function
------------------------------------------------------------------

If we left things at just the rewrite, then the lowering stage of the
compiler would fail, complaining it doesn't know how to lower
``arrayexpr`` operations.  We start by hooking a lowering function
into the target context whenever the :class:`RewriteArrayExprs` class
is instantiated by the compiler.  This hook causes the lowering pass to
call :func:`~numba.npyufunc.array_exprs._lower_array_expr` whenever it
encounters an ``arrayexr`` operator.

This function has two steps:

* Synthesize a Python function that implements the array expression:
  This new Python function essentially behaves like a Numpy
  :class:`~numpy.ufunc`, returning the result of the expression on
  scalar values in the broadcasted array arguments.  The lowering
  function accomplishes this by translating from the array expression
  tree into a Python AST.

* Compile the synthetic Python function into a kernel:  At this point,
  the lowering function relies on existing code for lowering ufunc and
  DUFunc kernels, calling
  :func:`numba.targets.numpyimpl.numpy_ufunc_kernel` after defining
  how to lower calls to the synthetic function.

The end result is similar to loop lifting in Numba's object mode.


Conclusions and Caveats
=======================

We have seen how to implement rewrites in Numba, starting with the
interface, and ending with an actual optimization.  The key points of
this section are:

* When writing a good plug-in, the matcher should try to get a
  go/no-go result as soon as possible.

* The rewrite application portion can be more computationally
  expensive, but should still generate code that won't cause infinite
  loops in the compiler.

* We use object state to communicate any results of matching to the
  rewrite application pass.
