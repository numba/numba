*************************************************
Chapter 4: Adding JIT and Optimizer Support
*************************************************

Written by `Chris Lattner <mailto:sabre@nondot.org>`_ and `Max
Shawabkeh <http://max99x.com>`_

Introduction # {#intro}
=======================

Welcome to Chapter 4 of the `Implementing a language with
LLVM <http://www.llvm.org/docs/tutorial/index.html>`_ tutorial. Chapters
1-3 described the implementation of a simple language and added support
for generating LLVM IR. This chapter describes two new techniques:
adding optimizer support to your language, and adding JIT compiler
support. These additions will demonstrate how to get nice, efficient
code for the Kaleidoscope language.

--------------

Trivial Constant Folding # {#trivialconstfold}
==============================================

Our demonstration for Chapter 3 is elegant and easy to extend.
Unfortunately, it does not produce wonderful code. The LLVM Builder,
however, does give us obvious optimizations when compiling simple code:

{% highlight bash %} ready> def test(x) 1+2+x Read function definition:
define double @test(double %x) { entry: %addtmp = fadd double
3.000000e+00, %x ret double %addtmp } {% endhighlight %}

This code is not a literal transcription of the AST built by parsing the
input. That would be:

{% highlight bash %} ready> def test(x) 1+2+x Read function definition:
define double @test(double %x) { entry: %addtmp = fadd double
2.000000e+00, 1.000000e+00 %addtmp1 = fadd double %addtmp, %x ret double
%addtmp1 } {% endhighlight %}

Constant folding, as seen above, in particular, is a very common and
very important optimization: so much so that many language implementors
implement constant folding support in their AST representation.

With LLVM, you don't need this support in the AST. Since all calls to
build LLVM IR go through the LLVM IR builder, the builder itself checked
to see if there was a constant folding opportunity when you call it. If
so, it just does the constant fold and return the constant instead of
creating an instruction.

Well, that was easy :). In practice, we recommend always using
``llvm.core.Builder`` when generating code like this. It has no
"syntactic overhead" for its use (you don't have to uglify your compiler
with constant checks everywhere) and it can dramatically reduce the
amount of LLVM IR that is generated in some cases (particular for
languages with a macro preprocessor or that use a lot of constants).

On the other hand, the ``Builder`` is limited by the fact that it does
all of its analysis inline with the code as it is built. If you take a
slightly more complex example:

{% highlight bash %} ready> def test(x) (1+2+x)\*(x+(1+2)) Read a
function definition: define double @test(double %x) { entry: %addtmp =
fadd double 3.000000e+00, %x ; [#uses=1] %addtmp1 = fadd double %x,
3.000000e+00 ; [#uses=1] %multmp = fmul double %addtmp, %addtmp1 ;
[#uses=1] ret double %multmp } {% endhighlight %}

In this case, the LHS and RHS of the multiplication are the same value.
We'd really like to see this generate"``tmp = x+3; result = tmp*tmp;``
instead of computing ``x+3`` twice.

Unfortunately, no amount of local analysis will be able to detect and
correct this. This requires two transformations: reassociation of
expressions (to make the add's lexically identical) and Common
Subexpression Elimination (CSE) to delete the redundant add instruction.
Fortunately, LLVM provides a broad range of optimizations that you can
use, in the form of "passes".

--------------

LLVM Optimization Passes # {#optimizerpasses}
=============================================

LLVM provides many optimization passes, which do many different sorts of
things and have different tradeoffs. Unlike other systems, LLVM doesn't
hold to the mistaken notion that one set of optimizations is right for
all languages and for all situations. LLVM allows a compiler implementor
to make complete decisions about what optimizations to use, in which
order, and in what situation.

As a concrete example, LLVM supports both "whole module" passes, which
look across as large of body of code as they can (often a whole file,
but if run at link time, this can be a substantial portion of the whole
program). It also supports and includes "per-function" passes which just
operate on a single function at a time, without looking at other
functions. For more information on passes and how they are run, see the
`How to Write a Pass <http://www.llvm.org/docs/WritingAnLLVMPass.html>`_
document and the `List of LLVM
Passes <http://www.llvm.org/docs/Passes.html>`_.

For Kaleidoscope, we are currently generating functions on the fly, one
at a time, as the user types them in. We aren't shooting for the
ultimate optimization experience in this setting, but we also want to
catch the easy and quick stuff where possible. As such, we will choose
to run a few per-function optimizations as the user types the function
in. If we wanted to make a "static Kaleidoscope compiler", we would use
exactly the code we have now, except that we would defer running the
optimizer until the entire file has been parsed.

In order to get per-function optimizations going, we need to set up a
`FunctionPassManager <http://www.llvm.org/docs/WritingAnLLVMPass.html#passmanager>`_
to hold and organize the LLVM optimizations that we want to run. Once we
have that, we can add a set of optimizations to run. The code looks like
this:

{% highlight python %} # The function optimization passes manager.
g\_llvm\_pass\_manager = FunctionPassManager.new(g\_llvm\_module)

The LLVM execution engine.
==========================

g\_llvm\_executor = ExecutionEngine.new(g\_llvm\_module)

...

def main(): # Set up the optimizer pipeline. Start with registering info
about how the # target lays out data structures.
g\_llvm\_pass\_manager.add(g\_llvm\_executor.target\_data) # Do simple
"peephole" optimizations and bit-twiddling optzns.
g\_llvm\_pass\_manager.add(PASS\_INSTRUCTION\_COMBINING) # Reassociate
expressions. g\_llvm\_pass\_manager.add(PASS\_REASSOCIATE) # Eliminate
Common SubExpressions. g\_llvm\_pass\_manager.add(PASS\_GVN) # Simplify
the control flow graph (deleting unreachable blocks, etc).
g\_llvm\_pass\_manager.add(PASS\_CFG\_SIMPLIFICATION)

g\_llvm\_pass\_manager.initialize() {% endhighlight %}

This code defines a ``FunctionPassManager``, ``g_llvm_pass_manager``.
Once it is set up, we use a series of "add" calls to add a bunch of LLVM
passes. The first pass is basically boilerplate, it adds a pass so that
later optimizations know how the data structures in the program are laid
out. (The "``g_llvm_executor``\ " variable is related to the JIT, which
we will get to in the next section.) In this case, we choose to add 4
optimization passes. The passes we chose here are a pretty standard set
of "cleanup" optimizations that are useful for a wide variety of code. I
won't delve into what they do but, believe me, they are a good starting
place :).

Once the pass manager is set up, we need to make use of it. We do this
by running it after our newly created function is constructed (in
``FunctionNode.CodeGen``), but before it is returned to the client:

{% highlight python %} return\_value = self.body.CodeGen()
g\_llvm\_builder.ret(return\_value)

::

      # Validate the generated code, checking for consistency.
      function.verify()

      # Optimize the function.
      g_llvm_pass_manager.run(function)

{% endhighlight %}

As you can see, this is pretty straightforward. The
``FunctionPassManager`` optimizes and updates the LLVM Function in
place, improving (hopefully) its body. With this in place, we can try
our test above again:

{% highlight bash %} ready> def test(x) (1+2+x)\*(x+(1+2)) Read a
function definition: define double @test(double %x) { entry: %addtmp =
fadd double %x, 3.000000e+00 ; [#uses=2] %multmp = fmul double %addtmp,
%addtmp ; [#uses=1] ret double %multmp } {% endhighlight %}

As expected, we now get our nicely optimized code, saving a floating
point add instruction from every execution of this function.

LLVM provides a wide variety of optimizations that can be used in
certain circumstances. Some `documentation about the various
passes <http://www.llvm.org/docs/Passes.html>`_ is available, but it
isn't very complete. Another good source of ideas can come from looking
at the passes that ``llvm-gcc`` or ``llvm-ld`` run to get started. The
``opt`` tool allows you to experiment with passes from the command line,
so you can see if they do anything.

Now that we have reasonable code coming out of our front-end, lets talk
about executing it!

--------------

Adding a JIT Compiler # {#jit}
==============================

Code that is available in LLVM IR can have a wide variety of tools
applied to it. For example, you can run optimizations on it (as we did
above), you can dump it out in textual or binary forms, you can compile
the code to an assembly file (.s) for some target, or you can JIT
compile it. The nice thing about the LLVM IR representation is that it
is the "common currency" between many different parts of the compiler.

In this section, we'll add JIT compiler support to our interpreter. The
basic idea that we want for Kaleidoscope is to have the user enter
function bodies as they do now, but immediately evaluate the top-level
expressions they type in. For example, if they type in "1 + 2", we
should evaluate and print out 3. If they define a function, they should
be able to call it from the command line.

In order to do this, we first declare and initialize the JIT. This is
done by adding and initializing a global variable:

{% highlight python %} # The LLVM execution engine. g\_llvm\_executor =
ExecutionEngine.new(g\_llvm\_module) {% endhighlight %}

This creates an abstract "Execution Engine" which can be either a JIT
compiler or the LLVM interpreter. LLVM will automatically pick a JIT
compiler for you if one is available for your platform, otherwise it
will fall back to the interpreter.

Once the ``ExecutionEngine`` is created, the JIT is ready to be used. We
can use the ``run_function`` method of the execution engine to execute a
compiled function and get its return value. In our case, this means that
we can change the code that parses a top-level expression to look like
this:

{% highlight python %} def HandleTopLevelExpression(self): try: function
= self.ParseTopLevelExpr().CodeGen() result =
g\_llvm\_executor.run\_function(function, []) print 'Evaluated to:',
result.as\_real(Type.double()) except Exception, e: print 'Error:', e
try: self.Next() # Skip for error recovery. except: pass {% endhighlight
%}

Recall that we compile top-level expressions into a self-contained LLVM
function that takes no arguments and returns the computed double.

With just these two changes, lets see how Kaleidoscope works now!

{% highlight python %} ready> 4+5 Read a top level expression: define
double @0() { entry: ret double 9.000000e+00 }

Evaluated to: 9.0 {% endhighlight %}

Well this looks like it is basically working. The dump of the function
shows the "no argument function that always returns double" that we
synthesize for each top-level expression that is typed in. This
demonstrates very basic functionality, but can we do more?

{% highlight python %} ready> def testfunc(x y) x + y\*2 Read a function
definition: define double @testfunc(double %x, double %y) { entry:
%multmp = fmul double %y, 2.000000e+00 ; [#uses=1] %addtmp = fadd double
%multmp, %x ; [#uses=1] ret double %addtmp }

ready> testfunc(4, 10) Read a top level expression: define double @0() {
entry: %calltmp = call double @testfunc(double 4.000000e+00, double
1.000000e+01) ; [#uses=1] ret double %calltmp }

*Evaluated to: 24.0* {% endhighlight %}

This illustrates that we can now call user code, but there is something
a bit subtle going on here. Note that we only invoke the JIT on the
anonymous functions that *call testfunc*, but we never invoked it on
*testfunc* itself. What actually happened here is that the JIT scanned
for all non-JIT'd functions transitively called from the anonymous
function and compiled all of them before returning from
``run_function()``.

The JIT provides a number of other more advanced interfaces for things
like freeing allocated machine code, rejit'ing functions to update them,
etc. However, even with this simple code, we get some surprisingly
powerful capabilities - check this out (I removed the dump of the
anonymous functions, you should get the idea by now :) :

{% highlight bash %} ready> extern sin(x) Read an extern: declare double
@sin(double)

ready> extern cos(x) Read an extern: declare double @cos(double)

ready> sin(1.0) *Evaluated to: 0.841470984808*

ready> def foo(x) sin(x)\ *sin(x) + cos(x)*\ cos(x) Read a function
definition: define double @foo(double %x) { entry: %calltmp = call
double @sin(double %x) ; [#uses=1] %calltmp1 = call double @sin(double
%x) ; [#uses=1] %multmp = fmul double %calltmp, %calltmp1 ; [#uses=1]
%calltmp2 = call double @cos(double %x) ; [#uses=1] %calltmp3 = call
double @cos(double %x) ; [#uses=1] %multmp4 = fmul double %calltmp2,
%calltmp3 ; [#uses=1] %addtmp = fadd double %multmp, %multmp4 ;
[#uses=1] ret double %addtmp }

ready> foo(4.0) *Evaluated to: 1.000000* {% endhighlight %}

Whoa, how does the JIT know about sin and cos? The answer is
surprisingly simple: in this example, the JIT started execution of a
function and got to a function call. It realized that the function was
not yet JIT compiled and invoked the standard set of routines to resolve
the function. In this case, there is no body defined for the function,
so the JIT ended up calling ``dlsym("sin")`` on the Python process that
is hosting our Kaleidoscope prompt. Since ``sin`` is defined within the
JIT's address space, it simply patches up calls in the module to call
the libm version of ``sin`` directly.

One interesting application of this is that we can now extend the
language by writing arbitrary C++ code to implement operations. For
example, we can create a C file with the following simple function:

{% highlight c %} #include

double putchard(double x) { putchar((char)x); return 0; } {%
endhighlight %}

We can then compile this into a shared library with GCC:

{% highlight bash %} gcc -shared -fPIC -o putchard.so putchard.c {%
endhighlight %}

Now we can load this library into the Python process using
``llvm.core.load_library_permanently`` and access it from Kaleidoscope
to produce simple output to the console:

{% highlight python %} >>> import llvm.core >>>
llvm.core.load\_library\_permanently('/home/max/llvmpy-tutorial/putchard.so')
>>> import kaleidoscope >>> kaleidoscope.main() ready> extern
putchard(x) Read an extern: declare double @putchard(double)

ready> putchard(65) + putchard(66) + putchard(67) + putchard(10) *ABC*
Evaluated to: 0.0 {% endhighlight %}

Similar code could be used to implement file I/O, console input, and
many other capabilities in Kaleidoscope.

This completes the JIT and optimizer chapter of the Kaleidoscope
tutorial. At this point, we can compile a non-Turing-complete
programming language, optimize and JIT compile it in a user-driven way.
Next up we'll look into `extending the language with control flow
constructs <PythonLangImpl5.html>`_, tackling some interesting LLVM IR
issues along the way.

--------------

Full Code Listing # {#code}
===========================

Here is the complete code listing for our running example, enhanced with
the LLVM JIT and optimizer:

{% highlight python %} #!/usr/bin/env python

import re from llvm.core import Module, Constant, Type, Function,
Builder, FCMP\_ULT from llvm.ee import ExecutionEngine, TargetData from
llvm.passes import FunctionPassManager from llvm.passes import
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

class EOFToken(object): pass

class DefToken(object): pass

class ExternToken(object): pass

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

      # If F took a different number of args, reject.
      if len(callee.args) != len(self.args):
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

# primary ::= identifierexpr \| numberexpr \| parenexpr def
ParsePrimary(self): if isinstance(self.current, IdentifierToken): return
self.ParseIdentifierExpr() elif isinstance(self.current, NumberToken):
return self.ParseNumberExpr() elif self.current == CharacterToken('('):
return self.ParseParenExpr() else: raise RuntimeError('Unknown token
when expecting an expression.')

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

**`Next: Extending the language: control flow <PythonLangImpl5.html>`_**
