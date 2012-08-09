*****************************************************
Chapter 5: Extending the Language: Control Flow
*****************************************************

Written by `Chris Lattner <mailto:sabre@nondot.org>`_ and `Max
Shawabkeh <http://max99x.com>`_

Introduction # {#intro}
=======================

Welcome to Chapter 5 of the `Implementing a language with
LLVM <http://www.llvm.org/docs/tutorial/index.html>`_ tutorial. Parts
1-4 described the implementation of the simple Kaleidoscope language and
included support for generating LLVM IR, followed by optimizations and a
JIT compiler. Unfortunately, as presented, Kaleidoscope is mostly
useless: it has no control flow other than call and return. This means
that you can't have conditional branches in the code, significantly
limiting its power. In this episode of "build that compiler", we'll
extend Kaleidoscope to have an if/then/else expression plus a simple
'for' loop.

--------------

If/Then/Else # {#ifthen}
========================

Extending Kaleidoscope to support if/then/else is quite straightforward.
It basically requires adding lexer support for this "new" concept to the
lexer, parser, AST, and LLVM code emitter. This example is nice, because
it shows how easy it is to "grow" a language over time, incrementally
extending it as new ideas are discovered.

Before we get going on "how" we add this extension, lets talk about
"what" we want. The basic idea is that we want to be able to write this
sort of thing:

{% highlight python %} def fib(x) if x < 3 then 1 else fib(x-1) +
fib(x-2) {% endhighlight %}

In Kaleidoscope, every construct is an expression: there are no
statements. As such, the if/then/else expression needs to return a value
like any other. Since we're using a mostly functional form, we'll have
it evaluate its conditional, then return the 'then' or 'else' value
based on how the condition was resolved. This is very similar to the C
"?:" expression.

The semantics of the if/then/else expression is that it evaluates the
condition to a boolean equality value: 0.0 is considered to be false and
everything else is considered to be true. If the condition is true, the
first subexpression is evaluated and returned, if the condition is
false, the second subexpression is evaluated and returned. Since
Kaleidoscope allows side-effects, this behavior is important to nail
down.

Now that we know what we "want", let's break this down into its
constituent pieces.

Lexer Extensions for If/Then/Else ## {#iflexer}
-----------------------------------------------

The lexer extensions are straightforward. First we add new token classes
for the relevant tokens:

{% highlight python %} class IfToken(object): pass class
ThenToken(object): pass class ElseToken(object): pass {% endhighlight %}

Once we have that, we recognize the new keywords in the lexer. This is
pretty simple stuff:

{% highlight python %} ... if identifier == 'def': yield DefToken() elif
identifier == 'extern': yield ExternToken() elif identifier == 'if':
yield IfToken() elif identifier == 'then': yield ThenToken() elif
identifier == 'else': yield ElseToken() else: yield
IdentifierToken(identifier) {% endhighlight %}

AST Extensions for If/Then/Else ## {#ifast}
-------------------------------------------

To represent the new expression we add a new AST node for it:

{% highlight python %} # Expression class for if/then/else. class
IfExpressionNode(ExpressionNode):

def **init**\ (self, condition, then\_branch, else\_branch):
self.condition = condition self.then\_branch = then\_branch
self.else\_branch = else\_branch

def CodeGen(self): ... {% endhighlight %}

The AST node just has pointers to the various subexpressions.

Parser Extensions for If/Then/Else ## {#ifparser}
-------------------------------------------------

Now that we have the relevant tokens coming from the lexer and we have
the AST node to build, our parsing logic is relatively straightforward.
First we define a new parsing function:

{% highlight python %} # ifexpr ::= 'if' expression 'then' expression
'else' expression def ParseIfExpr(self): self.Next() # eat the if.

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

{% endhighlight %}

Next we hook it up as a primary expression:

{% highlight python %} def ParsePrimary(self): if
isinstance(self.current, IdentifierToken): return
self.ParseIdentifierExpr() elif isinstance(self.current, NumberToken):
return self.ParseNumberExpr(); elif isinstance(self.current, IfToken):
return self.ParseIfExpr() elif self.current == CharacterToken('('):
return self.ParseParenExpr() else: raise RuntimeError('Unknown token
when expecting an expression.') {% endhighlight %}

LLVM IR for If/Then/Else ## {#ifir}
-----------------------------------

Now that we have it parsing and building the AST, the final piece is
adding LLVM code generation support. This is the most interesting part
of the if/then/else example, because this is where it starts to
introduce new concepts. All of the code above has been thoroughly
described in previous chapters.

To motivate the code we want to produce, lets take a look at a simple
example. Consider:

{% highlight python %} extern foo(); extern bar(); def baz(x) if x then
foo() else bar(); {% endhighlight %}

If you disable optimizations, the code you'll (soon) get from
Kaleidoscope looks something like this:

{% highlight llvm %} declare double @foo() declare double @bar() define
double @baz(double %x) { entry: %ifcond = fcmp one double %x,
0.000000e+00 br i1 %ifcond, label %then, label %else then: ; preds =
%entry %calltmp1 = call double @bar() else: ; preds = %entry %calltmp1 =
call double @bar() br label %ifcont ifcont: ; preds = %else, %then
%iftmp = phi double [ %calltmp, %then ], [ %calltmp1, %else ] ret double
%iftmp } {% endhighlight %}

To visualize the control flow graph, you can use a nifty feature of the
LLVM `opt <http://llvm.org/cmds/opt.html>`_ tool. If you put this LLVM
IR into "t.ll" and run ``llvm-as < t.ll | opt -analyze -view-cfg``, a
`window will pop
up <http://www.llvm.org/docs/ProgrammersManual.html#ViewGraph>`_ and
you'll see this graph:

Another way to get this is to call "``function.viewCFG()``\ " or
"``function.viewCFGOnly()``\ " (where F is a "``llvm.core.Function``\ ")
either by inserting actual calls into the code and recompiling or by
calling these in the debugger. LLVM has many nice features for
visualizing various graphs, but note that these are available only if
your LLVM was built with Graphviz support (accomplished by having
Graphviz and Ghostview installed when building LLVM).

Getting back to the generated code, it is fairly simple: the entry block
evaluates the conditional expression ("x" in our case here) and compares
the result to 0.0 with the
`fcmp <http://www.llvm.org/docs/LangRef.html#i_fcmp>`_ one instruction
('one' is "Ordered and Not Equal"). Based on the result of this
expression, the code jumps to either the "then" or "else" blocks, which
contain the expressions for the true/false cases.

Once the then/else blocks are finished executing, they both branch back
to the 'ifcont' block to execute the code that happens after the
if/then/else. In this case the only thing left to do is to return to the
caller of the function. The question then becomes: how does the code
know which expression to return?

The answer to this question involves an important SSA operation: the
`Phi
operation <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.
If you're not familiar with SSA, `the wikipedia
article <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
is a good introduction and there are various other introductions to it
available on your favorite search engine. The short version is that
"execution" of the Phi operation requires "remembering" which block
control came from. The Phi operation takes on the value corresponding to
the input control block. In this case, if control comes in from the
"then" block, it gets the value of "calltmp". If control comes from the
"else" block, it gets the value of "calltmp1".

At this point, you are probably starting to think "Oh no! This means my
simple and elegant front-end will have to start generating SSA form in
order to use LLVM!". Fortunately, this is not the case, and we strongly
advise *not* implementing an SSA construction algorithm in your
front-end unless there is an amazingly good reason to do so. In
practice, there are two sorts of values that float around in code
written for your average imperative programming language that might need
Phi nodes:

1. Code that involves user variables: ``x = 1; x = x + 1;``
2. Values that are implicit in the structure of your AST, such as the
   Phi node in this case.

In `Chapter 7 <PythonLangImpl7.html>`_ of this tutorial ("mutable
variables"), we'll talk about #1 in depth. For now, just believe me that
you don't need SSA construction to handle this case. For #2, you have
the choice of using the techniques that we will describe for #1, or you
can insert Phi nodes directly, if convenient. In this case, it is really
really easy to generate the Phi node, so we choose to do it directly.

Okay, enough of the motivation and overview, lets generate code!

Code Generation for If/Then/Else ## {#ifcodegen}
------------------------------------------------

In order to generate code for this, we implement the ``Codegen`` method
for ``IfExpressionNode``:

{% highlight python %} def CodeGen(self): condition =
self.condition.CodeGen()

::

    # Convert condition to a bool by comparing equal to 0.0.
    condition_bool = g_llvm_builder.fcmp(
        FCMP_ONE, condition, Constant.real(Type.double(), 0), 'ifcond')

{% endhighlight %}

This code is straightforward and similar to what we saw before. We emit
the expression for the condition, then compare that value to zero to get
a truth value as a 1-bit (bool) value.

{% highlight python %} function = g\_llvm\_builder.basic\_block.function

::

    # Create blocks for the then and else cases. Insert the 'then' block at the
    # end of the function.
    then_block = function.append_basic_block('then')
    else_block = function.append_basic_block('else')
    merge_block = function.append_basic_block('ifcond')

    g_llvm_builder.cbranch(condition_bool, then_block, else_block)

{% endhighlight %}

This code creates the basic blocks that are related to the if/then/else
statement, and correspond directly to the blocks in the example above.
The first line gets the current Function object that is being built. It
gets this by asking the builder for the current BasicBlock, and asking
that block for its "parent" (the function it is currently embedded
into).

Once it has that, it creates three block which are automatically
inserted into the end of the function. Once the blocks are created, we
can emit the conditional branch that chooses between them. Note that
creating new blocks does not implicitly affect the Builder, so it is
still inserting into the block that the condition went into.

{% highlight python %} # Emit then value.
g\_llvm\_builder.position\_at\_end(then\_block) then\_value =
self.then\_branch.CodeGen() g\_llvm\_builder.branch(merge\_block)

::

    # Codegen of 'Then' can change the current block; update then_block for the
    # PHI node.
    then_block = g_llvm_builder.basic_block

{% endhighlight %}

After the conditional branch is inserted, we move the builder to start
inserting into the "then" block. Strictly speaking, this call moves the
insertion point to be at the end of the specified block. However, since
the "then" block is empty, it also starts out by inserting at the
beginning of the block. :)

Once the insertion point is set, we recursively codegen the "then"
expression from the AST. To finish off the "then" block, we create an
unconditional branch to the merge block. One interesting (and very
important) aspect of the LLVM IR is that it `requires all basic blocks
to be
"terminated" <http://www.llvm.org/docs/LangRef.html#functionstructure>`_
with a `control flow
instruction <http://www.llvm.org/docs/LangRef.html#terminators>`_ such
as return or branch. This means that all control flow, *including
fallthroughs* must be made explicit in the LLVM IR. If you violate this
rule, the verifier will emit an error.

The final line here is quite subtle, but is very important. The basic
issue is that when we create the Phi node in the merge block, we need to
set up the block/value pairs that indicate how the Phi will work.
Importantly, the Phi node expects to have an entry for each predecessor
of the block in the CFG. Why then, are we getting the current block when
we just set it to then\_block 5 lines above? The problem is that the
"Then" expression may actually itself change the block that the Builder
is emitting into if, for example, it contains a nested "if/then/else"
expression. Because calling Codegen recursively could arbitrarily change
the notion of the current block, we are required to get an up-to-date
value for code that will set up the Phi node.

{% highlight python %} # Emit else block.
g\_llvm\_builder.position\_at\_end(else\_block) else\_value =
self.else\_branch.CodeGen() g\_llvm\_builder.branch(merge\_block)

::

    # Codegen of 'Else' can change the current block, update else_block for the
    # PHI node.
    else_block = g_llvm_builder.basic_block

{% endhighlight %}

Code generation for the 'else' block is basically identical to codegen
for the 'then' block. The only significant difference is the first line,
which adds the 'else' block to the function. Recall previously that the
'else' block was created, but not added to the function. Now that the
'then' and 'else' blocks are emitted, we can finish up with the merge
code:

{% highlight python %} # Emit merge block.
g\_llvm\_builder.position\_at\_end(merge\_block) phi =
g\_llvm\_builder.phi(Type.double(), 'iftmp')
phi.add\_incoming(then\_value, then\_block)
phi.add\_incoming(else\_value, else\_block)

::

    return phi

{% endhighlight %}

The first line changes the insertion point so that newly created code
will go into the "merge" block. Once that is done, we need to create the
PHI node and set up the block/value pairs for the PHI.

Finally, the CodeGen function returns the phi node as the value computed
by the if/then/else expression. In our example above, this returned
value will feed into the code for the top-level function, which will
create the return instruction.

Overall, we now have the ability to execute conditional code in
Kaleidoscope. With this extension, Kaleidoscope is a fairly complete
language that can calculate a wide variety of numeric functions. Next up
we'll add another useful expression that is familiar from non-functional
languages...

--------------

'for' Loop Expression # {#for}
==============================

Now that we know how to add basic control flow constructs to the
language, we have the tools to add more powerful things. Lets add
something more aggressive, a 'for' expression:

{% highlight python %} extern putchard(char) def printstar(n) for i = 1,
i < n, 1.0 in putchard(42) # ascii 42 = '\*'

::

    # print 100 '*' characters
    printstar(100)

{% endhighlight %}

This expression defines a new variable (``i`` in this case) which
iterates from a starting value, while the condition (``i < n`` in this
case) is true, incrementing by an optional step value ("1.0" in this
case). If the step value is omitted, it defaults to 1.0. While the loop
is true, it executes its body expression. Because we don't have anything
better to return, we'll just define the loop as always returning 0.0. In
the future when we have mutable variables, it will get more useful.

As before, lets talk about the changes that we need to Kaleidoscope to
support this.

Lexer Extensions for the 'for' Loop ## {#forlexer}
--------------------------------------------------

The lexer extensions are the same sort of thing as for if/then/else:

{% highlight python %} ...

class ThenToken(object): pass class ElseToken(object): pass class
ForToken(object): pass class InToken(object): pass

...

def Tokenize(string):

::

      ...

      elif identifier == 'else':
        yield ElseToken()
      elif identifier == 'for':
        yield ForToken()
      elif identifier == 'in':
        yield InToken()</b>
      else:
        yield IdentifierToken(identifier)

{% endhighlight %}

AST Extensions for the 'for' Loop ## {#forast}
----------------------------------------------

The AST node is just as simple. It basically boils down to capturing the
variable name and the constituent expressions in the node.

{% highlight python %} # Expression class for for/in. class
ForExpressionNode(ExpressionNode):

def **init**\ (self, loop\_variable, start, end, step, body):
self.loop\_variable = loop\_variable self.start = start self.end = end
self.step = step self.body = body

def CodeGen(self): ... {% endhighlight %}

Parser Extensions for the 'for' Loop ## {#forparser}
----------------------------------------------------

The parser code is also fairly standard. The only interesting thing here
is handling of the optional step value. The parser code handles it by
checking to see if the second comma is present. If not, it sets the step
value to null in the AST node:

{% highlight python %} # forexpr ::= 'for' identifier '=' expr ',' expr
(',' expr)? 'in' expression def ParseForExpr(self): self.Next() # eat
the for.

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

{% endhighlight %}

LLVM IR for the 'for' Loop ## {#forir}
--------------------------------------

Now we get to the good part: the LLVM IR we want to generate for this
thing. With the simple example above, we get this LLVM IR (note that
this dump is generated with optimizations disabled for clarity):

{% highlight llvm %} declare double @putchard(double) define double
@printstar(double %n) { entry: ; initial value = 1.0 (inlined into phi)
br label %loop loop: ; preds = %loop, %entry %i = phi double [
1.000000e+00, %entry ], [ %nextvar, %loop ] ; body %calltmp = call
double @putchard(double 4.200000e+01) ; increment %nextvar = fadd double
%i, 1.000000e+00 ; termination test %cmptmp = fcmp ult double %i, %n
%booltmp = uitofp i1 %cmptmp to double %loopcond = fcmp one double
%booltmp, 0.000000e+00 br i1 %loopcond, label %loop, label %afterloop
afterloop: ; preds = %loop ; loop always returns 0.0 ret double
0.000000e+00 } {% endhighlight %}

This loop contains all the same constructs we saw before: a phi node,
several expressions, and some basic blocks. Lets see how this fits
together.

Code Generation for the 'for' Loop ## {#forcodegen}
---------------------------------------------------

The first part of Codegen is very simple: we just output the start
expression for the loop value:

{% highlight python %} def CodeGen(self): # Emit the start code first,
without 'variable' in scope. start\_value = self.start.CodeGen() {%
endhighlight %}

With this out of the way, the next step is to set up the LLVM basic
block for the start of the loop body. In the case above, the whole loop
body is one block, but remember that the body code itself could consist
of multiple blocks (e.g. if it contains an if/then/else or a for/in
expression).

{% highlight python %} # Make the new basic block for the loop header,
inserting after current # block. function =
g\_llvm\_builder.basic\_block.function pre\_header\_block =
g\_llvm\_builder.basic\_block loop\_block =
function.append\_basic\_block('loop')

::

    # Insert an explicit fallthrough from the current block to the loop_block.
    g_llvm_builder.branch(loop_block)

{% endhighlight %}

This code is similar to what we saw for if/then/else. Because we will
need it to create the Phi node, we remember the block that falls through
into the loop. Once we have that, we create the actual block that starts
the loop and create an unconditional branch for the fall-through between
the two blocks.

{% highlight python %} # Start insertion in loop\_block.
g\_llvm\_builder.position\_at\_end(loop\_block);

::

    # Start the PHI node with an entry for start.
    variable_phi = g_llvm_builder.phi(Type.double(), self.loop_variable)
    variable_phi.add_incoming(start_value, pre_header_block)

{% endhighlight %}

Now that the "pre\_header\_block" for the loop is set up, we switch to
emitting code for the loop body. To begin with, we move the insertion
point and create the PHI node for the loop induction variable. Since we
already know the incoming value for the starting value, we add it to the
Phi node. Note that the Phi will eventually get a second value for the
backedge, but we can't set it up yet (because it doesn't exist!).

{% highlight python %} # Within the loop, the variable is defined equal
to the PHI node. If it # shadows an existing variable, we have to
restore it, so save it now. old\_value =
g\_named\_values.get(self.loop\_variable, None)
g\_named\_values[self.loop\_variable] = variable\_phi

::

    # Emit the body of the loop.  This, like any other expr, can change the
    # current BB.  Note that we ignore the value computed by the body.
    self.body.CodeGen()

{% endhighlight %}

Now the code starts to get more interesting. Our 'for' loop introduces a
new variable to the symbol table. This means that our symbol table can
now contain either function arguments or loop variables. To handle this,
before we codegen the body of the loop, we add the loop variable as the
current value for its name. Note that it is possible that there is a
variable of the same name in the outer scope. It would be easy to make
this an error (emit an error and return null if there is already an
entry for VarName) but we choose to allow shadowing of variables. In
order to handle this correctly, we remember the Value that we are
potentially shadowing in ``old_value`` (which will be None if there is
no shadowed variable).

Once the loop variable is set into the symbol table, the code
recursively codegen's the body. This allows the body to use the loop
variable: any references to it will naturally find it in the symbol
table.

{% highlight python %} # Emit the step value. if self.step: step\_value
= self.step.CodeGen() else: # If not specified, use 1.0. step\_value =
Constant.real(Type.double(), 1)

::

    next_value = g_llvm_builder.fadd(variable_phi, step_value, 'next')

{% endhighlight %}

Now that the body is emitted, we compute the next value of the iteration
variable by adding the step value, or 1.0 if it isn't present.
``next_value`` will be the value of the loop variable on the next
iteration of the loop.

{% highlight python %} # Compute the end condition and convert it to a
bool by comparing to 0.0. end\_condition = self.end.CodeGen()
end\_condition\_bool = g\_llvm\_builder.fcmp( FCMP\_ONE, end\_condition,
Constant.real(Type.double(), 0), 'loopcond') {% endhighlight %}

Finally, we evaluate the exit value of the loop, to determine whether
the loop should exit. This mirrors the condition evaluation for the
if/then/else statement.

{% highlight python %} # Create the "after loop" block and insert it.
loop\_end\_block = g\_llvm\_builder.basic\_block after\_block =
function.append\_basic\_block('afterloop')

::

    # Insert the conditional branch into the end of loop_end_block.
    g_llvm_builder.cbranch(end_condition_bool, loop_block, after_block)

    # Any new code will be inserted in after_block.
    g_llvm_builder.position_at_end(after_block)

{% endhighlight %}

With the code for the body of the loop complete, we just need to finish
up the control flow for it. This code remembers the end block (for the
phi node), then creates the block for the loop exit ("afterloop"). Based
on the value of the exit condition, it creates a conditional branch that
chooses between executing the loop again and exiting the loop. Any
future code is emitted in the "afterloop" block, so it sets the
insertion position to it.

{% highlight python %} # Add a new entry to the PHI node for the
backedge. variable\_phi.add\_incoming(next\_value, loop\_end\_block)

::

    # Restore the unshadowed variable.
    if old_value:
      g_named_values[self.loop_variable] = old_value
    else:
      del g_named_values[self.loop_variable]

    # for expr always returns 0.0.
    return Constant.real(Type.double(), 0)

{% endhighlight %}

The final code handles various cleanups: now that we have the
"next\_value", we can add the incoming value to the loop PHI node. After
that, we remove the loop variable from the symbol table, so that it
isn't in scope after the for loop. Finally, code generation of the for
loop always returns 0.0, so that is what we return from
``ForExpressionNode::CodeGen``.

With this, we conclude the "adding control flow to Kaleidoscope" chapter
of the tutorial. In this chapter we added two control flow constructs,
and used them to motivate a couple of aspects of the LLVM IR that are
important for front-end implementors to know. In the next chapter of our
saga, we will get a bit crazier and add `user-defined
operators <PythonLangImpl6.html>`_ to our poor innocent language.

--------------

Full Code Listing # {#code}
===========================

Here is the complete code listing for our running example, enhanced with
the if/then/else and for expressions:

{% highlight python %} #!/usr/bin/env python

import re from llvm.core import Module, Constant, Type, Function,
Builder from llvm.ee import ExecutionEngine, TargetData from llvm.passes
import FunctionPassManager

from llvm.core import FCMP\_ULT, FCMP\_ONE from llvm.passes import
(PASS\_INSTRUCTION\_COMBINING, PASS\_REASSOCIATE, PASS\_GVN,
PASS\_CFG\_SIMPLIFICATION)

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

Lexer
-----

The lexer yields one of these types for each token.
===================================================

class EOFToken(object): pass class DefToken(object): pass class
ExternToken(object): pass class IfToken(object): pass class
ThenToken(object): pass class ElseToken(object): pass class
ForToken(object): pass class InToken(object): pass

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
g\_named\_values[self.name] else: raise RuntimeError('Unknown variable
name: ' + self.name)

Expression class for a binary operator.
=======================================

class BinaryOperatorExpressionNode(ExpressionNode):

def **init**\ (self, operator, left, right): self.operator = operator
self.left = left self.right = right

def CodeGen(self): left = self.left.CodeGen() right =
self.right.CodeGen()

::

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
      raise RuntimeError('Unknown binary operator.')

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

def CodeGen(self): # Output this as: # ... # start = startexpr # goto
loop # loop: # variable = phi [start, loopheader], [nextvariable,
loopend] # ... # bodyexpr # ... # loopend: # step = stepexpr #
nextvariable = variable + step # endcond = endexpr # br endcond, loop,
endloop # outloop:

::

    # Emit the start code first, without 'variable' in scope.
    start_value = self.start.CodeGen()

    # Make the new basic block for the loop header, inserting after current
    # block.
    function = g_llvm_builder.basic_block.function
    pre_header_block = g_llvm_builder.basic_block
    loop_block = function.append_basic_block('loop')

    # Insert an explicit fallthrough from the current block to the loop_block.
    g_llvm_builder.branch(loop_block)

    # Start insertion in loop_block.
    g_llvm_builder.position_at_end(loop_block)

    # Start the PHI node with an entry for start.
    variable_phi = g_llvm_builder.phi(Type.double(), self.loop_variable)
    variable_phi.add_incoming(start_value, pre_header_block)

    # Within the loop, the variable is defined equal to the PHI node.  If it
    # shadows an existing variable, we have to restore it, so save it now.
    old_value = g_named_values.get(self.loop_variable, None)
    g_named_values[self.loop_variable] = variable_phi

    # Emit the body of the loop.  This, like any other expr, can change the
    # current BB.  Note that we ignore the value computed by the body.
    self.body.CodeGen()

    # Emit the step value.
    if self.step:
      step_value = self.step.CodeGen()
    else:
      # If not specified, use 1.0.
      step_value = Constant.real(Type.double(), 1)

    next_value = g_llvm_builder.fadd(variable_phi, step_value, 'next')

    # Compute the end condition and convert it to a bool by comparing to 0.0.
    end_condition = self.end.CodeGen()
    end_condition_bool = g_llvm_builder.fcmp(
        FCMP_ONE, end_condition, Constant.real(Type.double(), 0), 'loopcond')

    # Create the "after loop" block and insert it.
    loop_end_block = g_llvm_builder.basic_block
    after_block = function.append_basic_block('afterloop')

    # Insert the conditional branch into the end of loop_end_block.
    g_llvm_builder.cbranch(end_condition_bool, loop_block, after_block)

    # Any new code will be inserted in after_block.
    g_llvm_builder.position_at_end(after_block)

    # Add a new entry to the PHI node for the backedge.
    variable_phi.add_incoming(next_value, loop_end_block)

    # Restore the unshadowed variable.
    if old_value:
      g_named_values[self.loop_variable] = old_value
    else:
      del g_named_values[self.loop_variable]

    # for expr always returns 0.0.
    return Constant.real(Type.double(), 0)

This class represents the "prototype" for a function, which captures its name,
==============================================================================

and its argument names (thus implicitly the number of arguments the function
============================================================================

takes).
=======

class PrototypeNode(object):

def **init**\ (self, name, args): self.name = name self.args = args

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
      # Add arguments to variable symbol table.
      g_named_values[arg_name] = arg

    return function

This class represents a function definition itself.
===================================================

class FunctionNode(object):

def **init**\ (self, prototype, body): self.prototype = prototype
self.body = body

def CodeGen(self): # Clear scope. g\_named\_values.clear()

::

    # Create a function object.
    function = self.prototype.CodeGen()

    # Create a new basic block to start insertion into.
    block = function.append_basic_block('entry')
    global g_llvm_builder
    g_llvm_builder = Builder.new(block)

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
      raise

    return function

Parser
------

class Parser(object):

def **init**\ (self, tokens, binop\_precedence): self.tokens = tokens
self.binop\_precedence = binop\_precedence self.Next()

# Provide a simple token buffer. Parser.current is the current token the
# parser is looking at. Parser.Next() reads another token from the lexer
and # updates Parser.current with its results. def Next(self):
self.current = self.tokens.next()

# Gets the precedence of the current token, or -1 if the token is not a
binary # operator. def GetCurrentTokenPrecedence(self): if
isinstance(self.current, CharacterToken): return
self.binop\_precedence.get(self.current.char, -1) else: return -1

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

# primary ::= identifierexpr \| numberexpr \| parenexpr \| ifexpr \|
forexpr def ParsePrimary(self): if isinstance(self.current,
IdentifierToken): return self.ParseIdentifierExpr() elif
isinstance(self.current, NumberToken): return self.ParseNumberExpr()
elif isinstance(self.current, IfToken): return self.ParseIfExpr() elif
isinstance(self.current, ForToken): return self.ParseForExpr() elif
self.current == CharacterToken('('): return self.ParseParenExpr() else:
raise RuntimeError('Unknown token when expecting an expression.')

# binoprhs ::= (operator primary)\* def ParseBinOpRHS(self, left,
left\_precedence): # If this is a binary operator, find its precedence.
while True: precedence = self.GetCurrentTokenPrecedence()

::

      # If this is a binary operator that binds at least as tightly as the
      # current one, consume it; otherwise we are done.
      if precedence < left_precedence:
        return left

      binary_operator = self.current.char
      self.Next()  # eat the operator.

      # Parse the primary expression after the binary operator.
      right = self.ParsePrimary()

      # If binary_operator binds less tightly with right than the operator after
      # right, let the pending operator take right as its left.
      next_precedence = self.GetCurrentTokenPrecedence()
      if precedence < next_precedence:
        right = self.ParseBinOpRHS(right, precedence + 1)

      # Merge left/right.
      left = BinaryOperatorExpressionNode(binary_operator, left, right)

# expression ::= primary binoprhs def ParseExpression(self): left =
self.ParsePrimary() return self.ParseBinOpRHS(left, 0)

# prototype ::= id '(' id\* ')' def ParsePrototype(self): if not
isinstance(self.current, IdentifierToken): raise RuntimeError('Expected
function name in prototype.')

::

    function_name = self.current.name
    self.Next()  # eat function name.

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

    return PrototypeNode(function_name, arg_names)

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
result.as\_real(Type.double()) except Exception, e: print 'Error:', e
try: self.Next() # Skip for error recovery. except: pass

def Handle(self, function, message): try: print message,
function().CodeGen() except Exception, e: print 'Error:', e try:
self.Next() # Skip for error recovery. except: pass

Main driver code.
-----------------

def main(): # Set up the optimizer pipeline. Start with registering info
about how the # target lays out data structures.
g\_llvm\_pass\_manager.add(g\_llvm\_executor.target\_data) # Do simple
"peephole" optimizations and bit-twiddling optzns.
g\_llvm\_pass\_manager.add(PASS\_INSTRUCTION\_COMBINING) # Reassociate
expressions. g\_llvm\_pass\_manager.add(PASS\_REASSOCIATE) # Eliminate
Common SubExpressions. g\_llvm\_pass\_manager.add(PASS\_GVN) # Simplify
the control flow graph (deleting unreachable blocks, etc).
g\_llvm\_pass\_manager.add(PASS\_CFG\_SIMPLIFICATION)

g\_llvm\_pass\_manager.initialize()

# Install standard binary operators. # 1 is lowest possible precedence.
40 is the highest. operator\_precedence = { '<': 10, '+': 20, '-': 20,
'\*': 40 }

# Run the main "interpreter loop". while True: print 'ready>', try: raw
= raw\_input() except KeyboardInterrupt: break

::

    parser = Parser(Tokenize(raw), operator_precedence)
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

**`Next: Extending the language: user-defined
operators <PythonLangImpl6.html>`_**
