*******************************************************************************
Chapter 7: Extending the Language: Mutable Variables / SSA construction
*******************************************************************************

Written by `Chris Lattner <mailto:sabre@nondot.org>`_ and `Max
Shawabkeh <http://max99x.com>`_

Introduction # {#intro}
=======================

Welcome to Chapter 7 of the `Implementing a language with
LLVM <http://www.llvm.org/docs/tutorial/index.html>`_ tutorial. In
chapters 1 through 6, we've built a very respectable, albeit simple,
`functional programming
language <http://en.wikipedia.org/wiki/Functional_programming>`_. In our
journey, we learned some parsing techniques, how to build and represent
an AST, how to build LLVM IR, and how to optimize the resultant code as
well as JIT compile it.

While Kaleidoscope is interesting as a functional language, the fact
that it is functional makes it "too easy" to generate LLVM IR for it. In
particular, a functional language makes it very easy to build LLVM IR
directly in `SSA
form <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.
Since LLVM requires that the input code be in SSA form, this is a very
nice property and it is often unclear to newcomers how to generate code
for an imperative language with mutable variables.

The short (and happy) summary of this chapter is that there is no need
for your front-end to build SSA form: LLVM provides highly tuned and
well tested support for this, though the way it works is a bit
unexpected for some.

Why is this a hard problem? # {#why}
====================================

To understand why mutable variables cause complexities in SSA
construction, consider this extremely simple C example:

{% highlight python %} int G, H; int test(\_Bool Condition) { int X; if
(Condition) X = G; else X = H; return X; } {% endhighlight %}

In this case, we have the variable "X", whose value depends on the path
executed in the program. Because there are two different possible values
for X before the return instruction, a PHI node is inserted to merge the
two values. The LLVM IR that we want for this example looks like this:

{% highlight llvm %} @G = weak global i32 0 ; type of @G is i32\* @H =
weak global i32 0 ; type of @H is i32\* define i32 @test(i1 %Condition)
{ entry: br i1 %Condition, label %cond\_true, label %cond\_false
cond\_true: %X.0 = load i32\* @G br label %cond\_next cond\_false: %X.1
= load i32\* @H br label %cond\_next cond\_next: %X.2 = phi i32 [ %X.1,
%cond\_false ], [ %X.0, %cond\_true ] ret i32 %X.2 } {% endhighlight %}

In this example, the loads from the G and H global variables are
explicit in the LLVM IR, and they live in the then/else branches of the
if statement (cond\_true/cond\_false). In order to merge the incoming
values, the X.2 phi node in the cond\_next block selects the right value
to use based on where control flow is coming from: if control flow comes
from the cond\_false block, X.2 gets the value of X.1. Alternatively, if
control flow comes from cond\_true, it gets the value of X.0. The intent
of this chapter is not to explain the details of SSA form. For more
information, see one of the many `online
references <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.

The question for this article is "who places the phi nodes when lowering
assignments to mutable variables?". The issue here is that LLVM
*requires* that its IR be in SSA form: there is no "non-ssa" mode for
it. However, SSA construction requires non-trivial algorithms and data
structures, so it is inconvenient and wasteful for every front-end to
have to reproduce this logic.

Memory in LLVM # {#memory}
==========================

The 'trick' here is that while LLVM does require all register values to
be in SSA form, it does not require (or permit) memory objects to be in
SSA form. In the example above, note that the loads from G and H are
direct accesses to G and H: they are not renamed or versioned. This
differs from some other compiler systems, which do try to version memory
objects. In LLVM, instead of encoding dataflow analysis of memory into
the LLVM IR, it is handled with `Analysis
Passes <http://www.llvm.org/docs/WritingAnLLVMPass.html>`_ which are
computed on demand.

With this in mind, the high-level idea is that we want to make a stack
variable (which lives in memory, because it is on the stack) for each
mutable object in a function. To take advantage of this trick, we need
to talk about how LLVM represents stack variables.

In LLVM, all memory accesses are explicit with load/store instructions,
and it is carefully designed not to have (or need) an "address-of"
operator. Notice how the type of the @G/@H global variables is actually
"i32\ *" even though the variable is defined as "i32". What this means
is that @G defines*\ space\* for an i32 in the global data area, but its
*name* actually refers to the address for that space. Stack variables
work the same way, except that instead of being declared with global
variable definitions, they are declared with the `LLVM alloca
instruction <http://www.llvm.org/docs/LangRef.html#i_alloca>`_:

{% highlight python %} define i32 @example() { entry: %X = alloca i32 ;
type of %X is i32\ *. ... %tmp = load i32* %X ; load the stack value %X
from the stack. %tmp2 = add i32 %tmp, 1 ; increment it store i32 %tmp2,
i32\* %X ; store it back ... {% endhighlight %}

This code shows an example of how you can declare and manipulate a stack
variable in the LLVM IR. Stack memory allocated with the alloca
instruction is fully general: you can pass the address of the stack slot
to functions, you can store it in other variables, etc. In our example
above, we could rewrite the example to use the alloca technique to avoid
using a PHI node:

{% highlight llvm %} @G = weak global i32 0 ; type of @G is i32\* @H =
weak global i32 0 ; type of @H is i32\* define i32 @test(i1 %Condition)
{ entry: %X = alloca i32 ; type of %X is i32\ *. br i1 %Condition, label
%cond\_true, label %cond\_false cond\_true: %X.0 = load i32* @G store
i32 %X.0, i32\* %X ; Update X br label %cond\_next cond\_false: %X.1 =
load i32\* @H store i32 %X.1, i32\* %X ; Update X br label %cond\_next
cond\_next: %X.2 = load i32\* %X ; Read X ret i32 %X.2 } {% endhighlight
%}

With this, we have discovered a way to handle arbitrary mutable
variables without the need to create Phi nodes at all:

.. raw:: html

   <ol>
   <li>

Each mutable variable becomes a stack allocation.

.. raw:: html

   </li>
   <li>

Each read of the variable becomes a load from the stack.

.. raw:: html

   </li>
   <li>

Each update of the variable becomes a store to the stack.

.. raw:: html

   </li>
   <li>

Taking the address of a variable just uses the stack address directly.

.. raw:: html

   </li>
   </ol>

While this solution has solved our immediate problem, it introduced
another one: we have now apparently introduced a lot of stack traffic
for very simple and common operations, a major performance problem.
Fortunately for us, the LLVM optimizer has a highly-tuned optimization
pass named "mem2reg" that handles this case, promoting allocas like this
into SSA registers, inserting Phi nodes as appropriate. If you run this
example through the pass, for example, you'll get:

{% highlight bash %} $ llvm-as < example.ll \| opt -mem2reg \| llvm-dis
{% endhighlight %}

{% highlight llvm %} @G = weak global i32 0 @H = weak global i32 0
define i32 @test(i1 %Condition) { entry: br i1 %Condition, label
%cond\_true, label %cond\_false cond\_true: %X.0 = load i32\* @G br
label %cond\_next cond\_false: %X.1 = load i32\* @H br label %cond\_next
cond\_next: %X.01 = phi i32 [ %X.1, %cond\_false ], [ %X.0, %cond\_true
] ret i32 %X.01 } {% endhighlight %}

The mem2reg pass implements the standard "iterated dominance frontier"
algorithm for constructing SSA form and has a number of optimizations
that speed up (very common) degenerate cases. The mem2reg optimization
pass is the answer to dealing with mutable variables, and we highly
recommend that you depend on it. Note that mem2reg only works on
variables in certain circumstances:

-  mem2reg is alloca-driven: it looks for allocas and if it can handle
   them, it promotes them. It does not apply to global variables or heap
   allocations.

-  mem2reg only looks for alloca instructions in the entry block of the
   function. Being in the entry block guarantees that the alloca is only
   executed once, which makes analysis simpler.

-  mem2reg only promotes allocas whose uses are direct loads and stores.
   If the address of the stack object is passed to a function, or if any
   funny pointer arithmetic is involved, the alloca will not be
   promoted.

-  mem2reg only works on allocas of `first
   class <http://www.llvm.org/docs/LangRef.html#t_classifications>`_
   values (such as pointers, scalars and vectors), and only if the array
   size of the allocation is 1 (or missing in the .ll file). mem2reg is
   not capable of promoting structs or arrays to registers. Note that
   the "scalarrepl" pass is more powerful and can promote structs,
   "unions", and arrays in many cases.

All of these properties are easy to satisfy for most imperative
languages, and we'll illustrate it below with Kaleidoscope. The final
question you may be asking is: should I bother with this nonsense for my
front-end? Wouldn't it be better if I just did SSA construction
directly, avoiding use of the mem2reg optimization pass? In short, we
strongly recommend that you use this technique for building SSA form,
unless there is an extremely good reason not to. Using this technique
is:

-  Proven and well tested: llvm-gcc and clang both use this technique
   for local mutable variables. As such, the most common clients of LLVM
   are using this to handle a bulk of their variables. You can be sure
   that bugs are found fast and fixed early.

-  Extremely Fast: mem2reg has a number of special cases that make it
   fast in common cases as well as fully general. For example, it has
   fast-paths for variables that are only used in a single block,
   variables that only have one assignment point, good heuristics to
   avoid insertion of unneeded phi nodes, etc.

-  Needed for debug info generation: `Debug information in
   LLVM <http://www.llvm.org/docs/SourceLevelDebugging.html>`_ relies on
   having the address of the variable exposed so that debug info can be
   attached to it. This technique dovetails very naturally with this
   style of debug info.

If nothing else, this makes it much easier to get your front-end up and
running, and is very simple to implement. Lets extend Kaleidoscope with
mutable variables now!

--------------

Mutable Variables in Kaleidoscope # {#kalvars}
==============================================

Now that we know the sort of problem we want to tackle, lets see what
this looks like in the context of our little Kaleidoscope language.
We're going to add two features:

-  The ability to mutate variables with the '=' operator.
-  The ability to define new variables.

While the first item is really what this is about, we only have
variables for incoming arguments as well as for induction variables, and
redefining those only goes so far :). Also, the ability to define new
variables is a useful thing regardless of whether you will be mutating
them. Here's a motivating example that shows how we could use these:

{% highlight python %} # Define ':' for sequencing: as a low-precedence
operator that ignores operands # and just returns the RHS. def binary :
1 (x y) y;

Recursive fib, we could do this before.
=======================================

def fib(x) if (x < 3) then 1 else fib(x-1) + fib(x-2)

Iterative fib.
==============

def fibi(x) var a = 1, b = 1, c in (for i = 3, i < x in c = a + b : a =
b : b = c) : b

Call it.
========

fibi(10) {% endhighlight %}

In order to mutate variables, we have to change our existing variables
to use the "alloca trick". Once we have that, we'll add our new
operator, then extend Kaleidoscope to support new variable definitions.

--------------

Adjusting Existing Variables for Mutation # {#adjustments}
==========================================================

The symbol table in Kaleidoscope is managed at code generation time by
the ``g_named_values`` map. This map currently keeps track of the LLVM
"Value" that holds the double value for the named variable. In order to
support mutation, we need to change this slightly, so that it holds the
*memory location* of the variable in question. Note that this change is
a refactoring: it changes the structure of the code, but does not (by
itself) change the behavior of the compiler. All of these changes are
isolated in the Kaleidoscope code generator.

At this point in Kaleidoscope's development, it only supports variables
for two things: incoming arguments to functions and the induction
variable of 'for' loops. For consistency, we'll allow mutation of these
variables in addition to other user-defined variables. This means that
these will both need memory locations.

To start our transformation of Kaleidoscope, we will need to create the
allocas that we will store in ``g_named_values``. We'll use a helper
function that ensures that the allocas are created in the entry block of
the function:

{% highlight python %} # Creates an alloca instruction in the entry
block of the function. This is used # for mutable variables. def
CreateEntryBlockAlloca(function, var\_name): entry =
function.get\_entry\_basic\_block() builder = Builder.new(entry)
builder.position\_at\_beginning(entry) return
builder.alloca(Type.double(), var\_name) {% endhighlight %}

This code creates a temporary ``llvm.core.Builder`` that is pointing at
the first instruction of the entry block. It then creates an alloca with
the expected name and returns it. Because all values in Kaleidoscope are
doubles, there is no need to pass in a type to use.

With this in place, the first functionality change we want to make is to
variable references. In our new scheme, variables live on the stack, so
code generating a reference to them actually needs to produce a load
from the stack slot:

{% highlight python %} def CodeGen(self): if self.name in
g\_named\_values: return
g\_llvm\_builder.load(g\_named\_values[self.name], self.name) else:
raise RuntimeError('Unknown variable name: ' + self.name) {%
endhighlight %}

As you can see, this is pretty straightforward. Now we need to update
the things that define the variables to set up the alloca. We'll start
with ``ForExpressionNode.CodeGen`` (see the `full code listing <#code>`_
for the unabridged code):

{% highlight python %} def CodeGen(self): function =
g\_llvm\_builder.basic\_block.function

::

    # Create an alloca for the variable in the entry block.
    alloca = CreateEntryBlockAlloca(function, self.loop_variable)

    # Emit the start code first, without 'variable' in scope.
    start_value = self.start.CodeGen()

    # Store the value into the alloca.
    g_llvm_builder.store(start_value, alloca)
    ...
    # Compute the end condition.
    end_condition = self.end.CodeGen()

    # Reload, increment, and restore the alloca.  This handles the case where
    # the body of the loop mutates the variable.
    cur_value = g_llvm_builder.load(alloca, self.loop_variable)
    next_value = g_llvm_builder.fadd(cur_value, step_value, 'nextvar')
    g_llvm_builder.store(next_value, alloca)

    # Convert condition to a bool by comparing equal to 0.0.
    end_condition_bool = g_llvm_builder.fcmp(
        FCMP_ONE, end_condition, Constant.real(Type.double(), 0), 'loopcond')
    ...

{% endhighlight %}

This code is virtually identical to the code `before we allowed mutable
variables <PythonLangImpl5.html#forcodegen>`_. The big difference is
that we no longer have to construct a PHI node, and we use load/store to
access the variable as needed.

To support mutable argument variables, we need to also make allocas for
them. The code for this is also pretty simple:

{% highlight python %} class PrototypeNode(object): ... # Create an
alloca for each argument and register the argument in the symbol # table
so that references to it will succeed. def CreateArgumentAllocas(self,
function): for arg\_name, arg in zip(self.args, function.args): alloca =
CreateEntryBlockAlloca(function, arg\_name) g\_llvm\_builder.store(arg,
alloca) g\_named\_values[arg\_name] = alloca {% endhighlight %}

For each argument, we make an alloca, store the input value to the
function into the alloca, and register the alloca as the memory location
for the argument. This method gets invoked by ``FunctionNode.CodeGen``
right after it sets up the entry block for the function.

The final missing piece is adding the mem2reg pass, which allows us to
get good codegen once again:

{% highlight python %} from llvm.passes import
(PASS\_PROMOTE\_MEMORY\_TO\_REGISTER, PASS\_INSTRUCTION\_COMBINING,
PASS\_REASSOCIATE, PASS\_GVN, PASS\_CFG\_SIMPLIFICATION) ... def main():
# Set up the optimizer pipeline. Start with registering info about how
the # target lays out data structures.
g\_llvm\_pass\_manager.add(g\_llvm\_executor.target\_data) # Promote
allocas to registers.
g\_llvm\_pass\_manager.add(PASS\_PROMOTE\_MEMORY\_TO\_REGISTER) # Do
simple "peephole" optimizations and bit-twiddling optzns.
g\_llvm\_pass\_manager.add(PASS\_INSTRUCTION\_COMBINING) # Reassociate
expressions. g\_llvm\_pass\_manager.add(PASS\_REASSOCIATE) {%
endhighlight %}

It is interesting to see what the code looks like before and after the
mem2reg optimization runs. For example, this is the before/after code
for our recursive fib function. Before the optimization:

{% highlight llvm %} define double @fib(double %x) { entry: %x1 = alloca
double store double %x, double\* %x1 %x2 = load double\* %x1 %cmptmp =
fcmp ult double %x2, 3.000000e+00 %booltmp = uitofp i1 %cmptmp to double
%ifcond = fcmp one double %booltmp, 0.000000e+00 br i1 %ifcond, label
%then, label %else then: ; preds = %entry br label %ifcont else: ; preds
= %entry %x3 = load double\* %x1 %subtmp = fsub double %x3, 1.000000e+00
%calltmp = call double @fib(double %subtmp) %x4 = load double\* %x1
%subtmp5 = fsub double %x4, 2.000000e+00 %calltmp6 = call double
@fib(double %subtmp5) %addtmp = fadd double %calltmp, %calltmp6 br label
%ifcont ifcont: ; preds = %else, %then %iftmp = phi double [
1.000000e+00, %then ], [ %addtmp, %else ] ret double %iftmp } {%
endhighlight %}

Here there is only one variable (x, the input argument) but you can
still see the extremely simple-minded code generation strategy we are
using. In the entry block, an alloca is created, and the initial input
value is stored into it. Each reference to the variable does a reload
from the stack. Also, note that we didn't modify the if/then/else
expression, so it still inserts a PHI node. While we could make an
alloca for it, it is actually easier to create a PHI node for it, so we
still just make the PHI.

Here is the code after the mem2reg pass runs:

{% highlight llvm %} define double @fib(double %x) { entry: %cmptmp =
fcmp ult double %x, 3.000000e+00 %booltmp = uitofp i1 %cmptmp to double
%ifcond = fcmp one double %booltmp, 0.000000e+00 br i1 %ifcond, label
%then, label %else then: br label %ifcont else: %subtmp = fsub double
%x, 1.000000e+00 %calltmp = call double @fib(double %subtmp) %subtmp5 =
fsub double %x, 2.000000e+00 %calltmp6 = call double @fib(double
%subtmp5) %addtmp = fadd double %calltmp, %calltmp6 br label %ifcont
ifcont: ; preds = %else, %then %iftmp = phi double [ 1.000000e+00, %then
], [ %addtmp, %else ] ret double %iftmp } {% endhighlight %}

This is a trivial case for mem2reg, since there are no redefinitions of
the variable. The point of showing this is to calm your tension about
inserting such blatent inefficiencies :).

After the rest of the optimizers run, we get:

{% highlight llvm %} define double @fib(double %x) { entry: %cmptmp =
fcmp ult double %x, 3.000000e+00 %booltmp = uitofp i1 %cmptmp to double
%ifcond = fcmp ueq double %booltmp, 0.000000e+00 br i1 %ifcond, label
%else, label %ifcont else: %subtmp = fsub double %x, 1.000000e+00
%calltmp = call double @fib(double %subtmp) %subtmp5 = fsub double %x,
2.000000e+00 %calltmp6 = call double @fib(double %subtmp5) %addtmp =
fadd double %calltmp, %calltmp6 ret double %addtmp ifcont: ret double
1.000000e+00 } {% endhighlight %}

Here we see that the simplifycfg pass decided to clone the return
instruction into the end of the 'else' block. This allowed it to
eliminate some branches and the PHI node.

Now that all symbol table references are updated to use stack variables,
we'll add the assignment operator.

--------------

New Assignment Operator # {#assignment}
=======================================

With our current framework, adding a new assignment operator is really
simple. We will parse it just like any other binary operator, but handle
it internally (instead of allowing the user to define it). The first
step is to set a precedence:

{% highlight python %} def main(): ... # Install standard binary
operators. # 1 is lowest possible precedence. 40 is the highest.
g\_binop\_precedence['='] = 2 g\_binop\_precedence['<'] = 10
g\_binop\_precedence['+'] = 20 g\_binop\_precedence['-'] = 20 {%
endhighlight %}

Now that the parser knows the precedence of the binary operator, it
takes care of all the parsing and AST generation. We just need to
implement codegen for the assignment operator. This looks like:

{% highlight python %} class
BinaryOperatorExpressionNode(ExpressionNode): ... def CodeGen(self): # A
special case for '=' because we don't want to emit the LHS as an #
expression. if self.operator == '=': # Assignment requires the LHS to be
an identifier. if not isinstance(self.left, VariableExpressionNode):
raise RuntimeError('Destination of "=" must be a variable.') {%
endhighlight %}

Unlike the rest of the binary operators, our assignment operator doesn't
follow the "emit LHS, emit RHS, do computation" model. As such, it is
handled as a special case before the other binary operators are handled.
The other strange thing is that it requires the LHS to be a variable. It
is invalid to have ``(x+1) = expr`` -- only things like ``x = expr`` are
allowed.

{% highlight python %} # Codegen the RHS. value = self.right.CodeGen()

::

      # Look up the name.
      variable = g_named_values[self.left.name]

      # Store the value and return it.
      g_llvm_builder.store(value, variable)

      return value
    ...

{% endhighlight %}

Once we have the variable, CodeGening the assignment is straightforward:
we emit the RHS of the assignment, create a store, and return the
computed value. Returning a value allows for chained assignments like
``X = (Y = Z)``.

Now that we have an assignment operator, we can mutate loop variables
and arguments. For example, we can now run code like this:

{% highlight python %} # Function to print a double. extern printd(x)

Define ':' for sequencing: as a low-precedence operator that ignores operands
=============================================================================

and just returns the RHS.
=========================

def binary : 1 (x y) y

def test(x) printd(x) : x = 4 : printd(x)

test(123) {% endhighlight %}

When run, this example prints "123" and then "4", showing that we did
actually mutate the value! Okay, we have now officially implemented our
goal: getting this to work requires SSA construction in the general
case. However, to be really useful, we want the ability to define our
own local variables. Let's add this next!

--------------

User-defined Local Variables # {#localvars}
===========================================

Adding var/in is just like any other other extensions we made to
Kaleidoscope: we extend the lexer, the parser, the AST and the code
generator. The first step for adding our new 'var/in' construct is to
extend the lexer. As before, this is pretty trivial, the code looks like
this:

{% highlight python %} ... class UnaryToken(object): pass class
VarToken(object): pass ... def Tokenize(string): ... elif identifier ==
'unary': yield UnaryToken() elif identifier == 'var': yield VarToken()
else: yield IdentifierToken(identifier) {% endhighlight %}

The next step is to define the AST node that we will construct. For
var/in, it looks like this:

{% highlight python %} # Expression class for var/in. class
VarExpressionNode(ExpressionNode):

def **init**\ (self, variables, body): self.variables = variables
self.body = body

def CodeGen(self): ... {% endhighlight %}

var/in allows a list of names to be defined all at once, and each name
can optionally have an initializer value. As such, we capture this
information in the variables list. Also, var/in has a body, this body is
allowed to access the variables defined by the var/in.

With this in place, we can define the parser pieces. The first thing we
do is add it as a primary expression:

{% highlight python %} # primary ::= # dentifierexpr \| numberexpr \|
parenexpr \| ifexpr \| forexpr \| varexpr def ParsePrimary(self): if
isinstance(self.current, IdentifierToken): return
self.ParseIdentifierExpr() elif isinstance(self.current, NumberToken):
return self.ParseNumberExpr() elif isinstance(self.current, IfToken):
return self.ParseIfExpr() elif isinstance(self.current, ForToken):
return self.ParseForExpr() elif isinstance(self.current, VarToken):
return self.ParseVarExpr() elif self.current == CharacterToken('('):
return self.ParseParenExpr() else: raise RuntimeError('Unknown token
when expecting an expression.') {% endhighlight %}

Next we define ParseVarExpr:

{% highlight python %} # varexpr ::= 'var' (identifier ('='
expression)?)+ 'in' expression def ParseVarExpr(self): self.Next() # eat
'var'.

::

    variables = {}

    # At least one variable name is required.
    if not isinstance(self.current, IdentifierToken):
      raise RuntimeError('Expected identifier after "var".')

{% endhighlight %}

The first part of this code parses the list of identifier/expr pairs
into the local ``variables`` list.

{% highlight python %} while True: var\_name = self.current.name
self.Next() # eat the identifier.

::

      # Read the optional initializer.
      if self.current == CharacterToken('='):
        self.Next()  # eat '='.
        variables[var_name] = self.ParseExpression()
      else:
        variables[var_name] = None

      # End of var list, exit loop.
      if self.current != CharacterToken(','):
        break
      self.Next()  # eat ','.

      if not isinstance(self.current, IdentifierToken):
        raise RuntimeError('Expected identifier after "," in a var expression.')

{% endhighlight %}

Once all the variables are parsed, we then parse the body and create the
AST node:

{% highlight python %} # At this point, we have to have 'in'. if not
isinstance(self.current, InToken): raise RuntimeError('Expected "in"
keyword after "var".') self.Next() # eat 'in'.

::

    body = self.ParseExpression()

    return VarExpressionNode(variables, body)

{% endhighlight %}

Now that we can parse and represent the code, we need to support
emission of LLVM IR for it. This code starts out with:

{% highlight python %} class VarExpressionNode(ExpressionNode): ... def
CodeGen(self): old\_bindings = {} function =
g\_llvm\_builder.basic\_block.function

::

    # Register all variables and emit their initializer.
    for var_name, var_expression in self.variables.iteritems():
      # Emit the initializer before adding the variable to scope, this prevents
      # the initializer from referencing the variable itself, and permits stuff
      # like this:
      #  var a = 1 in
      #    var a = a in ...   # refers to outer 'a'.
      if var_expression is not None:
        var_value = var_expression.CodeGen()
      else:
        var_value = Constant.real(Type.double(), 0)

      alloca = CreateEntryBlockAlloca(function, var_name)
      g_llvm_builder.store(var_value, alloca)

      # Remember the old variable binding so that we can restore the binding
      # when we unrecurse.
      old_bindings[var_name] = g_named_values.get(var_name, None)

      # Remember this binding.
      g_named_values[var_name] = alloca

{% endhighlight %}

Basically it loops over all the variables, installing them one at a
time. For each variable we put into the symbol table, we remember the
previous value that we replace in ``old_bindings``.

There are more comments here than code. The basic idea is that we emit
the initializer, create the alloca, then update the symbol table to
point to it. Once all the variables are installed in the symbol table,
we evaluate the body of the var/in expression:

{% highlight python %} # Codegen the body, now that all vars are in
scope. body = self.body.CodeGen() {% endhighlight %}

Finally, before returning, we restore the previous variable bindings:

{% highlight python %} # Pop all our variables from scope. for var\_name
in self.variables: if old\_bindings[var\_name] is not None:
g\_named\_values[var\_name] = old\_bindings[var\_name] else: del
g\_named\_values[var\_name]

::

    # Return the body computation.
    return body

{% endhighlight %}

The end result of all of this is that we get properly scoped variable
definitions, and we even (trivially) allow mutation of them :).

With this, we completed what we set out to do. Our nice iterative fib
example from the intro compiles and runs just fine. The mem2reg pass
optimizes all of our stack variables into SSA registers, inserting PHI
nodes where needed, and our front-end remains simple: no "iterated
dominance frontier" computation anywhere in sight.

--------------

Full Code Listing # {#code}
===========================

Here is the complete code listing for our running example, enhanced with
mutable variables and var/in support:

{% highlight python %} #!/usr/bin/env python

import re from llvm.core import Module, Constant, Type, Function,
Builder from llvm.ee import ExecutionEngine, TargetData from llvm.passes
import FunctionPassManager

from llvm.core import FCMP\_ULT, FCMP\_ONE from llvm.passes import
(PASS\_PROMOTE\_MEMORY\_TO\_REGISTER, PASS\_INSTRUCTION\_COMBINING,
PASS\_REASSOCIATE, PASS\_GVN, PASS\_CFG\_SIMPLIFICATION)

Globals
-------

The LLVM module, which holds all the IR code.
=============================================

g\_llvm\_module = Module.new('my cool jit')

The LLVM instruction builder. Created whenever a new function is entered.
=========================================================================

g\_llvm\_builder = None

A dictionary that keeps track of which values are defined in the current scope
==============================================================================

and what their LLVM representation is.
======================================

g\_named\_values = {}

The function optimization passes manager.
=========================================

g\_llvm\_pass\_manager = FunctionPassManager.new(g\_llvm\_module)

The LLVM execution engine.
==========================

g\_llvm\_executor = ExecutionEngine.new(g\_llvm\_module)

The binary operator precedence chart.
=====================================

g\_binop\_precedence = {}

Creates an alloca instruction in the entry block of the function. This is used
==============================================================================

for mutable variables.
======================

def CreateEntryBlockAlloca(function, var\_name): entry =
function.get\_entry\_basic\_block() builder = Builder.new(entry)
builder.position\_at\_beginning(entry) return
builder.alloca(Type.double(), var\_name)

Lexer
-----

The lexer yields one of these types for each token.
===================================================

class EOFToken(object): pass class DefToken(object): pass class
ExternToken(object): pass class IfToken(object): pass class
ThenToken(object): pass class ElseToken(object): pass class
ForToken(object): pass class InToken(object): pass class
BinaryToken(object): pass class UnaryToken(object): pass class
VarToken(object): pass

class IdentifierToken(object): def **init**\ (self, name): self.name =
name

class NumberToken(object): def **init**\ (self, value): self.value =
value

class CharacterToken(object): def **init**\ (self, char): self.char =
char def **eq**\ (self, other): return isinstance(other, CharacterToken)
and self.char == other.char def **ne**\ (self, other): return not self
== other

Regular expressions that tokens and comments of our language.
=============================================================

REGEX\_NUMBER = re.compile('[0-9]+(?:.[0-9]+)?') REGEX\_IDENTIFIER =
re.compile('[a-zA-Z][a-zA-Z0-9]\ *') REGEX\_COMMENT = re.compile('#.*')

def Tokenize(string): while string: # Skip whitespace. if
string[0].isspace(): string = string[1:] continue

::

    # Run regexes.
    comment_match = REGEX_COMMENT.match(string)
    number_match = REGEX_NUMBER.match(string)
    identifier_match = REGEX_IDENTIFIER.match(string)

    # Check if any of the regexes matched and yield the appropriate result.
    if comment_match:
      comment = comment_match.group(0)
      string = string[len(comment):]
    elif number_match:
      number = number_match.group(0)
      yield NumberToken(float(number))
      string = string[len(number):]
    elif identifier_match:
      identifier = identifier_match.group(0)
      # Check if we matched a keyword.
      if identifier == 'def':
        yield DefToken()
      elif identifier == 'extern':
        yield ExternToken()
      elif identifier == 'if':
        yield IfToken()
      elif identifier == 'then':
        yield ThenToken()
      elif identifier == 'else':
        yield ElseToken()
      elif identifier == 'for':
        yield ForToken()
      elif identifier == 'in':
        yield InToken()
      elif identifier == 'binary':
        yield BinaryToken()
      elif identifier == 'unary':
        yield UnaryToken()
      elif identifier == 'var':
        yield VarToken()
      else:
        yield IdentifierToken(identifier)
      string = string[len(identifier):]
    else:
      # Yield the ASCII value of the unknown character.
      yield CharacterToken(string[0])
      string = string[1:]

yield EOFToken()

Abstract Syntax Tree (aka Parse Tree)
-------------------------------------

Base class for all expression nodes.
====================================

class ExpressionNode(object): pass

Expression class for numeric literals like "1.0".
=================================================

class NumberExpressionNode(ExpressionNode):

def **init**\ (self, value): self.value = value

def CodeGen(self): return Constant.real(Type.double(), self.value)

Expression class for referencing a variable, like "a".
======================================================

class VariableExpressionNode(ExpressionNode):

def **init**\ (self, name): self.name = name

def CodeGen(self): if self.name in g\_named\_values: return
g\_llvm\_builder.load(g\_named\_values[self.name], self.name) else:
raise RuntimeError('Unknown variable name: ' + self.name)

Expression class for a binary operator.
=======================================

class BinaryOperatorExpressionNode(ExpressionNode):

def **init**\ (self, operator, left, right): self.operator = operator
self.left = left self.right = right

def CodeGen(self): # A special case for '=' because we don't want to
emit the LHS as an # expression. if self.operator == '=': # Assignment
requires the LHS to be an identifier. if not isinstance(self.left,
VariableExpressionNode): raise RuntimeError('Destination of "=" must be
a variable.')

::

      # Codegen the RHS.
      value = self.right.CodeGen()

      # Look up the name.
      variable = g_named_values[self.left.name]

      # Store the value and return it.
      g_llvm_builder.store(value, variable)

      return value

    left = self.left.CodeGen()
    right = self.right.CodeGen()

    if self.operator == '+':
      return g_llvm_builder.fadd(left, right, 'addtmp')
    elif self.operator == '-':
      return g_llvm_builder.fsub(left, right, 'subtmp')
    elif self.operator == '*':
      return g_llvm_builder.fmul(left, right, 'multmp')
    elif self.operator == '<':
      result = g_llvm_builder.fcmp(FCMP_ULT, left, right, 'cmptmp')
      # Convert bool 0 or 1 to double 0.0 or 1.0.
      return g_llvm_builder.uitofp(result, Type.double(), 'booltmp')
    else:
      function = g_llvm_module.get_function_named('binary' + self.operator)
      return g_llvm_builder.call(function, [left, right], 'binop')

Expression class for function calls.
====================================

class CallExpressionNode(ExpressionNode):

def **init**\ (self, callee, args): self.callee = callee self.args =
args

def CodeGen(self): # Look up the name in the global module table. callee
= g\_llvm\_module.get\_function\_named(self.callee)

::

    # Check for argument mismatch error.
    if len(callee.args) != len(self.args):
      raise RuntimeError('Incorrect number of arguments passed.')

    arg_values = [i.CodeGen() for i in self.args]

    return g_llvm_builder.call(callee, arg_values, 'calltmp')

Expression class for if/then/else.
==================================

class IfExpressionNode(ExpressionNode):

def **init**\ (self, condition, then\_branch, else\_branch):
self.condition = condition self.then\_branch = then\_branch
self.else\_branch = else\_branch

def CodeGen(self): condition = self.condition.CodeGen()

::

    # Convert condition to a bool by comparing equal to 0.0.
    condition_bool = g_llvm_builder.fcmp(
        FCMP_ONE, condition, Constant.real(Type.double(), 0), 'ifcond')

    function = g_llvm_builder.basic_block.function

    # Create blocks for the then and else cases. Insert the 'then' block at the
    # end of the function.
    then_block = function.append_basic_block('then')
    else_block = function.append_basic_block('else')
    merge_block = function.append_basic_block('ifcond')

    g_llvm_builder.cbranch(condition_bool, then_block, else_block)

    # Emit then value.
    g_llvm_builder.position_at_end(then_block)
    then_value = self.then_branch.CodeGen()
    g_llvm_builder.branch(merge_block)

    # Codegen of 'Then' can change the current block; update then_block for the
    # PHI node.
    then_block = g_llvm_builder.basic_block

    # Emit else block.
    g_llvm_builder.position_at_end(else_block)
    else_value = self.else_branch.CodeGen()
    g_llvm_builder.branch(merge_block)

    # Codegen of 'Else' can change the current block, update else_block for the
    # PHI node.
    else_block = g_llvm_builder.basic_block

    # Emit merge block.
    g_llvm_builder.position_at_end(merge_block)
    phi = g_llvm_builder.phi(Type.double(), 'iftmp')
    phi.add_incoming(then_value, then_block)
    phi.add_incoming(else_value, else_block)

    return phi

Expression class for for/in.
============================

class ForExpressionNode(ExpressionNode):

def **init**\ (self, loop\_variable, start, end, step, body):
self.loop\_variable = loop\_variable self.start = start self.end = end
self.step = step self.body = body

def CodeGen(self): # Output this as: # var = alloca double # ... # start
= startexpr # store start -> var # goto loop # loop: # ... # bodyexpr #
... # loopend: # step = stepexpr # endcond = endexpr # # curvar = load
var # nextvar = curvar + step # store nextvar -> var # br endcond, loop,
endloop # outloop:

::

    function = g_llvm_builder.basic_block.function

    # Create an alloca for the variable in the entry block.
    alloca = CreateEntryBlockAlloca(function, self.loop_variable)

    # Emit the start code first, without 'variable' in scope.
    start_value = self.start.CodeGen()

    # Store the value into the alloca.
    g_llvm_builder.store(start_value, alloca)

    # Make the new basic block for the loop, inserting after current block.
    loop_block = function.append_basic_block('loop')

    # Insert an explicit fall through from the current block to the loop_block.
    g_llvm_builder.branch(loop_block)

    # Start insertion in loop_block.
    g_llvm_builder.position_at_end(loop_block)

    # Within the loop, the variable is defined equal to the alloca.  If it
    # shadows an existing variable, we have to restore it, so save it now.
    old_value = g_named_values.get(self.loop_variable, None)
    g_named_values[self.loop_variable] = alloca

    # Emit the body of the loop.  This, like any other expr, can change the
    # current BB.  Note that we ignore the value computed by the body.
    self.body.CodeGen()

    # Emit the step value.
    if self.step:
      step_value = self.step.CodeGen()
    else:
      # If not specified, use 1.0.
      step_value = Constant.real(Type.double(), 1)

    # Compute the end condition.
    end_condition = self.end.CodeGen()

    # Reload, increment, and restore the alloca.  This handles the case where
    # the body of the loop mutates the variable.
    cur_value = g_llvm_builder.load(alloca, self.loop_variable)
    next_value = g_llvm_builder.fadd(cur_value, step_value, 'nextvar')
    g_llvm_builder.store(next_value, alloca)

    # Convert condition to a bool by comparing equal to 0.0.
    end_condition_bool = g_llvm_builder.fcmp(
        FCMP_ONE, end_condition, Constant.real(Type.double(), 0), 'loopcond')

    # Create the "after loop" block and insert it.
    after_block = function.append_basic_block('afterloop')

    # Insert the conditional branch into the end of loop_block.
    g_llvm_builder.cbranch(end_condition_bool, loop_block, after_block)

    # Any new code will be inserted in after_block.
    g_llvm_builder.position_at_end(after_block)

    # Restore the unshadowed variable.
    if old_value is not None:
      g_named_values[self.loop_variable] = old_value
    else:
      del g_named_values[self.loop_variable]

    # for expr always returns 0.0.
    return Constant.real(Type.double(), 0)

Expression class for a unary operator.
======================================

class UnaryExpressionNode(ExpressionNode):

def **init**\ (self, operator, operand): self.operator = operator
self.operand = operand

def CodeGen(self): operand = self.operand.CodeGen() function =
g\_llvm\_module.get\_function\_named('unary' + self.operator) return
g\_llvm\_builder.call(function, [operand], 'unop')

Expression class for var/in.
============================

class VarExpressionNode(ExpressionNode):

def **init**\ (self, variables, body): self.variables = variables
self.body = body

def CodeGen(self): old\_bindings = {} function =
g\_llvm\_builder.basic\_block.function

::

    # Register all variables and emit their initializer.
    for var_name, var_expression in self.variables.iteritems():
      # Emit the initializer before adding the variable to scope, this prevents
      # the initializer from referencing the variable itself, and permits stuff
      # like this:
      #  var a = 1 in
      #    var a = a in ...   # refers to outer 'a'.
      if var_expression is not None:
        var_value = var_expression.CodeGen()
      else:
        var_value = Constant.real(Type.double(), 0)

      alloca = CreateEntryBlockAlloca(function, var_name)
      g_llvm_builder.store(var_value, alloca)

      # Remember the old variable binding so that we can restore the binding
      # when we unrecurse.
      old_bindings[var_name] = g_named_values.get(var_name, None)

      # Remember this binding.
      g_named_values[var_name] = alloca

    # Codegen the body, now that all vars are in scope.
    body = self.body.CodeGen()

    # Pop all our variables from scope.
    for var_name in self.variables:
      if old_bindings[var_name] is not None:
        g_named_values[var_name] = old_bindings[var_name]
      else:
        del g_named_values[var_name]

    # Return the body computation.
    return body

This class represents the "prototype" for a function, which captures its name,
==============================================================================

and its argument names (thus implicitly the number of arguments the function
============================================================================

takes), as well as if it is an operator.
========================================

class PrototypeNode(object):

def **init**\ (self, name, args, is\_operator=False, precedence=0):
self.name = name self.args = args self.is\_operator = is\_operator
self.precedence = precedence

def IsBinaryOp(self): return self.is\_operator and len(self.args) == 2

def GetOperatorName(self): assert self.is\_operator return self.name[-1]

def CodeGen(self): # Make the function type, eg. double(double,double).
funct\_type = Type.function( Type.double(), [Type.double()] \*
len(self.args), False)

::

    function = Function.new(g_llvm_module, funct_type, self.name)

    # If the name conflicted, there was already something with the same name.
    # If it has a body, don't allow redefinition or reextern.
    if function.name != self.name:
      function.delete()
      function = g_llvm_module.get_function_named(self.name)

      # If the function already has a body, reject this.
      if not function.is_declaration:
        raise RuntimeError('Redefinition of function.')

      # If the function took a different number of args, reject.
      if len(function.args) != len(self.args):
        raise RuntimeError('Redeclaration of a function with different number '
                           'of args.')

    # Set names for all arguments and add them to the variables symbol table.
    for arg, arg_name in zip(function.args, self.args):
      arg.name = arg_name

    return function

# Create an alloca for each argument and register the argument in the
symbol # table so that references to it will succeed. def
CreateArgumentAllocas(self, function): for arg\_name, arg in
zip(self.args, function.args): alloca = CreateEntryBlockAlloca(function,
arg\_name) g\_llvm\_builder.store(arg, alloca)
g\_named\_values[arg\_name] = alloca

This class represents a function definition itself.
===================================================

class FunctionNode(object):

def **init**\ (self, prototype, body): self.prototype = prototype
self.body = body

def CodeGen(self): # Clear scope. g\_named\_values.clear()

::

    # Create a function object.
    function = self.prototype.CodeGen()

    # If this is a binary operator, install its precedence.
    if self.prototype.IsBinaryOp():
      operator = self.prototype.GetOperatorName()
      g_binop_precedence[operator] = self.prototype.precedence

    # Create a new basic block to start insertion into.
    block = function.append_basic_block('entry')
    global g_llvm_builder
    g_llvm_builder = Builder.new(block)

    # Add all arguments to the symbol table and create their allocas.
    self.prototype.CreateArgumentAllocas(function)

    # Finish off the function.
    try:
      return_value = self.body.CodeGen()
      g_llvm_builder.ret(return_value)

      # Validate the generated code, checking for consistency.
      function.verify()

      # Optimize the function.
      g_llvm_pass_manager.run(function)
    except:
      function.delete()
      if self.prototype.IsBinaryOp():
        del g_binop_precedence[self.prototype.GetOperatorName()]
      raise

    return function

Parser
------

class Parser(object):

def **init**\ (self, tokens): self.tokens = tokens self.Next()

# Provide a simple token buffer. Parser.current is the current token the
# parser is looking at. Parser.Next() reads another token from the lexer
and # updates Parser.current with its results. def Next(self):
self.current = self.tokens.next()

# Gets the precedence of the current token, or -1 if the token is not a
binary # operator. def GetCurrentTokenPrecedence(self): if
isinstance(self.current, CharacterToken): return
g\_binop\_precedence.get(self.current.char, -1) else: return -1

# identifierexpr ::= identifier \| identifier '(' expression\* ')' def
ParseIdentifierExpr(self): identifier\_name = self.current.name
self.Next() # eat identifier.

::

    if self.current != CharacterToken('('):  # Simple variable reference.
      return VariableExpressionNode(identifier_name)

    # Call.
    self.Next()  # eat '('.
    args = []
    if self.current != CharacterToken(')'):
      while True:
        args.append(self.ParseExpression())
        if self.current == CharacterToken(')'):
          break
        elif self.current != CharacterToken(','):
          raise RuntimeError('Expected ")" or "," in argument list.')
        self.Next()

    self.Next()  # eat ')'.
    return CallExpressionNode(identifier_name, args)

# numberexpr ::= number def ParseNumberExpr(self): result =
NumberExpressionNode(self.current.value) self.Next() # consume the
number. return result

# parenexpr ::= '(' expression ')' def ParseParenExpr(self): self.Next()
# eat '('.

::

    contents = self.ParseExpression()

    if self.current != CharacterToken(')'):
      raise RuntimeError('Expected ")".')
    self.Next()  # eat ')'.

    return contents

# ifexpr ::= 'if' expression 'then' expression 'else' expression def
ParseIfExpr(self): self.Next() # eat the if.

::

    # condition.
    condition = self.ParseExpression()

    if not isinstance(self.current, ThenToken):
      raise RuntimeError('Expected "then".')
    self.Next()  # eat the then.

    then_branch = self.ParseExpression()

    if not isinstance(self.current, ElseToken):
      raise RuntimeError('Expected "else".')
    self.Next()  # eat the else.

    else_branch = self.ParseExpression()

    return IfExpressionNode(condition, then_branch, else_branch)

# forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in'
expression def ParseForExpr(self): self.Next() # eat the for.

::

    if not isinstance(self.current, IdentifierToken):
      raise RuntimeError('Expected identifier after for.')

    loop_variable = self.current.name
    self.Next()  # eat the identifier.

    if self.current != CharacterToken('='):
      raise RuntimeError('Expected "=" after for variable.')
    self.Next()  # eat the '='.

    start = self.ParseExpression()

    if self.current != CharacterToken(','):
      raise RuntimeError('Expected "," after for start value.')
    self.Next()  # eat the ','.

    end = self.ParseExpression()

    # The step value is optional.
    if self.current == CharacterToken(','):
      self.Next()  # eat the ','.
      step = self.ParseExpression()
    else:
      step = None

    if not isinstance(self.current, InToken):
      raise RuntimeError('Expected "in" after for variable specification.')
    self.Next()  # eat 'in'.

    body = self.ParseExpression()

    return ForExpressionNode(loop_variable, start, end, step, body)

# varexpr ::= 'var' (identifier ('=' expression)?)+ 'in' expression def
ParseVarExpr(self): self.Next() # eat 'var'.

::

    variables = {}

    # At least one variable name is required.
    if not isinstance(self.current, IdentifierToken):
      raise RuntimeError('Expected identifier after "var".')

    while True:
      var_name = self.current.name
      self.Next()  # eat the identifier.

      # Read the optional initializer.
      if self.current == CharacterToken('='):
        self.Next()  # eat '='.
        variables[var_name] = self.ParseExpression()
      else:
        variables[var_name] = None

      # End of var list, exit loop.
      if self.current != CharacterToken(','):
        break
      self.Next()  # eat ','.

      if not isinstance(self.current, IdentifierToken):
        raise RuntimeError('Expected identifier after "," in a var expression.')

    # At this point, we have to have 'in'.
    if not isinstance(self.current, InToken):
      raise RuntimeError('Expected "in" keyword after "var".')
    self.Next()  # eat 'in'.

    body = self.ParseExpression()

    return VarExpressionNode(variables, body)

# primary ::= # dentifierexpr \| numberexpr \| parenexpr \| ifexpr \|
forexpr \| varexpr def ParsePrimary(self): if isinstance(self.current,
IdentifierToken): return self.ParseIdentifierExpr() elif
isinstance(self.current, NumberToken): return self.ParseNumberExpr()
elif isinstance(self.current, IfToken): return self.ParseIfExpr() elif
isinstance(self.current, ForToken): return self.ParseForExpr() elif
isinstance(self.current, VarToken): return self.ParseVarExpr() elif
self.current == CharacterToken('('): return self.ParseParenExpr() else:
raise RuntimeError('Unknown token when expecting an expression.')

# unary ::= primary \| unary\_operator unary def ParseUnary(self): # If
the current token is not an operator, it must be a primary expression.
if (not isinstance(self.current, CharacterToken) or self.current in
[CharacterToken('('), CharacterToken(',')]): return self.ParsePrimary()

::

    # If this is a unary operator, read it.
    operator = self.current.char
    self.Next()  # eat the operator.
    return UnaryExpressionNode(operator, self.ParseUnary())

# binoprhs ::= (binary\_operator unary)\* def ParseBinOpRHS(self, left,
left\_precedence): # If this is a binary operator, find its precedence.
while True: precedence = self.GetCurrentTokenPrecedence()

::

      # If this is a binary operator that binds at least as tightly as the
      # current one, consume it; otherwise we are done.
      if precedence < left_precedence:
        return left

      binary_operator = self.current.char
      self.Next()  # eat the operator.

      # Parse the unary expression after the binary operator.
      right = self.ParseUnary()

      # If binary_operator binds less tightly with right than the operator after
      # right, let the pending operator take right as its left.
      next_precedence = self.GetCurrentTokenPrecedence()
      if precedence < next_precedence:
        right = self.ParseBinOpRHS(right, precedence + 1)

      # Merge left/right.
      left = BinaryOperatorExpressionNode(binary_operator, left, right)

# expression ::= unary binoprhs def ParseExpression(self): left =
self.ParseUnary() return self.ParseBinOpRHS(left, 0)

# prototype # ::= id '(' id\* ')' # ::= binary LETTER number? (id, id) #
::= unary LETTER (id) def ParsePrototype(self): precedence = None if
isinstance(self.current, IdentifierToken): kind = 'normal'
function\_name = self.current.name self.Next() # eat function name. elif
isinstance(self.current, UnaryToken): kind = 'unary' self.Next() # eat
'unary'. if not isinstance(self.current, CharacterToken): raise
RuntimeError('Expected an operator after "unary".') function\_name =
'unary' + self.current.char self.Next() # eat the operator. elif
isinstance(self.current, BinaryToken): kind = 'binary' self.Next() # eat
'binary'. if not isinstance(self.current, CharacterToken): raise
RuntimeError('Expected an operator after "binary".') function\_name =
'binary' + self.current.char self.Next() # eat the operator. if
isinstance(self.current, NumberToken): if not 1 <= self.current.value <=
100: raise RuntimeError('Invalid precedence: must be in range [1,
100].') precedence = self.current.value self.Next() # eat the
precedence. else: raise RuntimeError('Expected function name, "unary" or
"binary" in ' 'prototype.')

::

    if self.current != CharacterToken('('):
      raise RuntimeError('Expected "(" in prototype.')
    self.Next()  # eat '('.

    arg_names = []
    while isinstance(self.current, IdentifierToken):
      arg_names.append(self.current.name)
      self.Next()

    if self.current != CharacterToken(')'):
      raise RuntimeError('Expected ")" in prototype.')

    # Success.
    self.Next()  # eat ')'.

    if kind == 'unary' and len(arg_names) != 1:
      raise RuntimeError('Invalid number of arguments for a unary operator.')
    elif kind == 'binary' and len(arg_names) != 2:
      raise RuntimeError('Invalid number of arguments for a binary operator.')

    return PrototypeNode(function_name, arg_names, kind != 'normal', precedence)

# definition ::= 'def' prototype expression def ParseDefinition(self):
self.Next() # eat def. proto = self.ParsePrototype() body =
self.ParseExpression() return FunctionNode(proto, body)

# toplevelexpr ::= expression def ParseTopLevelExpr(self): proto =
PrototypeNode('', []) return FunctionNode(proto, self.ParseExpression())

# external ::= 'extern' prototype def ParseExtern(self): self.Next() #
eat extern. return self.ParsePrototype()

# Top-Level parsing def HandleDefinition(self):
self.Handle(self.ParseDefinition, 'Read a function definition:')

def HandleExtern(self): self.Handle(self.ParseExtern, 'Read an extern:')

def HandleTopLevelExpression(self): try: function =
self.ParseTopLevelExpr().CodeGen() result =
g\_llvm\_executor.run\_function(function, []) print 'Evaluated to:',
result.as\_real(Type.double()) except Exception, e: raise#print
'Error:', e try: self.Next() # Skip for error recovery. except: pass

def Handle(self, function, message): try: print message,
function().CodeGen() except Exception, e: raise#print 'Error:', e try:
self.Next() # Skip for error recovery. except: pass

Main driver code.
-----------------

def main(): # Set up the optimizer pipeline. Start with registering info
about how the # target lays out data structures.
g\_llvm\_pass\_manager.add(g\_llvm\_executor.target\_data) # Promote
allocas to registers.
g\_llvm\_pass\_manager.add(PASS\_PROMOTE\_MEMORY\_TO\_REGISTER) # Do
simple "peephole" optimizations and bit-twiddling optzns.
g\_llvm\_pass\_manager.add(PASS\_INSTRUCTION\_COMBINING) # Reassociate
expressions. g\_llvm\_pass\_manager.add(PASS\_REASSOCIATE) # Eliminate
Common SubExpressions. g\_llvm\_pass\_manager.add(PASS\_GVN) # Simplify
the control flow graph (deleting unreachable blocks, etc).
g\_llvm\_pass\_manager.add(PASS\_CFG\_SIMPLIFICATION)

g\_llvm\_pass\_manager.initialize()

# Install standard binary operators. # 1 is lowest possible precedence.
40 is the highest. g\_binop\_precedence['='] = 2
g\_binop\_precedence['<'] = 10 g\_binop\_precedence['+'] = 20
g\_binop\_precedence['-'] = 20 g\_binop\_precedence['\*'] = 40

# Run the main "interpreter loop". while True: print 'ready<', try: raw
= raw\_input() except KeyboardInterrupt: break

::

    parser = Parser(Tokenize(raw))
    while True:
      # top ::= definition | external | expression | EOF
      if isinstance(parser.current, EOFToken):
        break
      if isinstance(parser.current, DefToken):
        parser.HandleDefinition()
      elif isinstance(parser.current, ExternToken):
        parser.HandleExtern()
      else:
        parser.HandleTopLevelExpression()

# Print out all of the generated code. print '', g\_llvm\_module

if **name** == '**main**\ ': main() {% endhighlight %}

--------------

**`Next: Conclusion and other useful LLVM
tidbits <PythonLangImpl8.html>`_**
