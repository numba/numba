*******************************************
Chapter 3: Code generation to LLVM IR
*******************************************

Written by `Chris Lattner <mailto:sabre@nondot.org>`_ and `Max
Shawabkeh <http://max99x.com>`_

Introduction # {#intro}
=======================

Welcome to Chapter 3 of the `Implementing a language with
LLVM <http://www.llvm.org/docs/tutorial/index.html>`_ tutorial. This
chapter shows you how to transform the `Abstract Syntax
Tree <PythonLangImpl2.html>`_, built in Chapter 2, into LLVM IR. This
will teach you a little bit about how LLVM does things, as well as
demonstrate how easy it is to use. It's much more work to build a lexer
and parser than it is to generate LLVM IR code. :)

**Please note**: the code in this chapter and later requires llvmpy 0.6
and LLVM 2.7. Earlier versions will most likely not work with it. Also
note that you need to use a version of this tutorial that matches your
llvmpy release: If you are using an official llvmpy release, use the
version of the documentation on the `llvmpy examples
page <http://www.mdevan.org/llvmpy/examples.html>`_

--------------

Code Generation Setup # {#basics}
=================================

In order to generate LLVM IR, we want some simple setup to get started.
First we define code generation methods in each AST node class:

{% highlight python %} # Expression class for numeric literals like
"1.0". class NumberExpressionNode(ExpressionNode):

def **init**\ (self, value): self.value = value

def CodeGen(self): ...

Expression class for referencing a variable, like "a".
======================================================

class VariableExpressionNode(ExpressionNode):

def **init**\ (self, name): self.name = name

def CodeGen(self): ...

... {% endhighlight %}

The ``CodeGen`` method says to emit IR for that AST node along with all
the things it depends on, and they all return an LLVM Value object.
"Value" is the class used to represent a "`Static Single Assignment
(SSA) <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
register" or "SSA value" in LLVM. The most distinct aspect of SSA values
is that their value is computed as the related instruction executes, and
it does not get a new value until (and if) the instruction re-executes.
In other words, there is no way to "change" an SSA value. For more
information, please read up on `Static Single
Assignment <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
- the concepts are really quite natural once you grok them.

We will also need to define some global variables which we will be used
during code generation:

{% highlight python %} # The LLVM module, which holds all the IR code.
g\_llvm\_module = Module.new('my cool jit')

The LLVM instruction builder. Created whenever a new function is entered.
=========================================================================

g\_llvm\_builder = None

A dictionary that keeps track of which values are defined in the current scope
==============================================================================

and what their LLVM representation is.
======================================

g\_named\_values = {} {% endhighlight %}

``g_llvm_module`` is the LLVM construct that contains all of the
functions and global variables in a chunk of code. In many ways, it is
the top-level structure that the LLVM IR uses to contain code.

``g_llvm_builder`` is a helper object that makes it easy to generate
LLVM instructions. Instances of the
`llvm.core.Builder <llvm.core.Builder.html>`_ class keep track of the
current place to insert instructions and have methods to create new
instructions. Note that we do not initialize this variable; instead, it
will be initialized whenever we start generating code for a function.

Finally, ``g_named_values`` is a dictionary that keeps track of which
values are defined in the current scope and what their LLVM
representation is. In other words, it is a symbol table for the code. In
this form of Kaleidoscope, the only things that can be referenced are
function parameters. As such, function parameters will be in this map
when generating code for their function body.

With these basics in place, we can start talking about how to generate
code for each expression. Note that this assumes that ``g_llvm_builder``
has been set up to generate code *into* something. For now, we'll assume
that this has already been done, and we'll just use it to emit code.

--------------

Expression Code Generation # {#exprs}
=====================================

Generating LLVM code for expression nodes is very straightforward: less
than 35 lines of commented code for all four of our expression nodes.
First we'll do numeric literals:

{% highlight python %} def CodeGen(self): return
Constant.real(Type.double(), self.value) {% endhighlight %}

In llvmpy, floating point numeric constants are represented with the
``llvm.core.ConstantFP`` class. To create one, we can use the static
``real()`` method in the ``llvm.core.Constant`` class. This code
basically just creates and returns a ``ConstantFP``. Note that in the
LLVM IR constants are all uniqued together and shared. For this reason,
we create the constant through a factory method instead of instantiating
one directly.

{% highlight python %} def CodeGen(self): if self.name in
g\_named\_values: return g\_named\_values[self.name] else: raise
RuntimeError('Unknown variable name: ' + self.name) {% endhighlight %}

References to variables are also quite simple using LLVM. In the simple
version of Kaleidoscope, we assume that the variable has already been
emitted somewhere and its value is available. In practice, the only
values that can be in the ``g_named_values`` dictionary are function
arguments. This code simply checks to see that the specified name is in
the map (if not, an unknown variable is being referenced) and returns
the value for it. In future chapters, we'll add support for `loop
induction variables <PythonLangImpl5.html#for>`_ in the symbol table,
and for `local variables <PythonLangImpl7.html#localvars>`_.

{% highlight python %} def CodeGen(self): left = self.left.CodeGen()
right = self.right.CodeGen()

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

{% endhighlight %}

Binary operators start to get more interesting. The basic idea here is
that we recursively emit code for the left-hand side of the expression,
then the right-hand side, then we compute the result of the binary
expression depending on which operator is being used.

In the example above, the LLVM builder class is starting to show its
value. ``g_llvm_builder`` knows where to insert the newly created
instruction, all you have to do is specify what instruction to create
(e.g. with ``add``), which operands to use (``left`` and ``right`` here)
and optionally provide a name for the generated instruction.

One nice thing about LLVM is that the name is just a hint. For instance,
if the code above emits multiple "addtmp" variables, LLVM will
automatically provide each one with an increasing, unique numeric
suffix. Local value names for instructions are purely optional, but it
makes it much easier to read the IR dumps.

`LLVM instructions <http://www.llvm.org/docs/LangRef.html#instref>`_ are
constrained by strict rules: for example, the Left and Right operators
of an `add instruction <http://www.llvm.org/docs/LangRef.html#i_add>`_
must have the same type, and the result type of the add must match the
operand types. Because all values in Kaleidoscope are doubles, this
makes for very simple code for add, sub and mul.

On the other hand, LLVM specifies that the `fcmp
instruction <http://www.llvm.org/docs/LangRef.html#i_fcmp>`_ always
returns an 'i1' value (a one bit integer). The problem with this is that
Kaleidoscope wants the value to be a 0.0 or 1.0 value. In order to get
these semantics, we combine the fcmp instruction with a `uitofp
instruction <http://www.llvm.org/docs/LangRef.html#i_uitofp>`_. This
instruction converts its input integer into a floating point value by
treating the input as an unsigned value. In contrast, if we used the
`sitofp instruction <http://www.llvm.org/docs/LangRef.html#i_sitofp>`_,
the Kaleidoscope ``<`` operator would return 0.0 and -1.0, depending on
the input value.

{% highlight python %} def CodeGen(self): # Look up the name in the
global module table. callee =
g\_llvm\_module.get\_function\_named(self.callee)

::

    # Check for argument mismatch error.
    if len(callee.args) != len(self.args):
      raise RuntimeError('Incorrect number of arguments passed.')

    arg_values = [i.CodeGen() for i in self.args]

    return g_llvm_builder.call(callee, arg_values, 'calltmp')

{% endhighlight %}

Code generation for function calls is quite straightforward with LLVM.
The code above initially does a function name lookup in the LLVM
Module's symbol table. Recall that the LLVM Module is the container that
holds all of the functions we are JIT'ing. By giving each function the
same name as what the user specifies, we can use the LLVM symbol table
to resolve function names for us.

Once we have the function to call, we codegen each argument that is to
be passed in, and create an LLVM `call
instruction <http://www.llvm.org/docs/LangRef.html#i_call>`_. Note that
LLVM uses the native C calling conventions by default, allowing these
calls to also call into standard library functions like "sin" and "cos",
with no additional effort.

This wraps up our handling of the four basic expressions that we have so
far in Kaleidoscope. Feel free to go in and add some more. For example,
by browsing the `LLVM language
reference <http://www.llvm.org/docs/LangRef.html>`_ you'll find several
other interesting instructions that are really easy to plug into our
basic framework.

--------------

Function Code Generation # {#funcs}
===================================

Code generation for prototypes and functions must handle a number of
details, which make their code less beautiful than expression code
generation, but allows us to illustrate some important points. First,
let's talk about code generation for prototypes: they are used both for
function bodies and external function declarations. The code starts
with:

{% highlight python %} def CodeGen(self): # Make the function type, eg.
double(double,double). funct\_type = Type.function( Type.double(),
[Type.double()] \* len(self.args), False)

::

    function = Function.new(g_llvm_module, funct_type, self.name)

{% endhighlight %}

The call to ``Type.function`` creates the ``FunctionType`` that should
be used for a given Prototype. Since all function arguments in
Kaleidoscope are of type double, the first line creates a list of "N"
LLVM double types. It then uses the ``Type.function`` method to create a
function type that takes "N" doubles as arguments, returns one double as
a result, and that is not vararg (the False parameter indicates this).
Note that Types in LLVM are uniqued just like Constants are, so you
don't instantiate them directly.

The final line above actually creates the function that the prototype
will correspond to. This indicates the type and name to use, as well as
which module to insert into. Note that by default, the function will
have `external
linkage <<http://www.llvm.org/docs/LangRef.html#linkage>`_, which means
that the function may be defined outside the current module and/or that
it is callable by functions outside the module. The name passed in is
the name the user specified: since ``g_llvm_module`` is specified, this
name is registered in ``g_llvm_module``'s symbol table, which is used by
the function call code above.

{% highlight python %} # If the name conflicted, there was already
something with the same name. # If it has a body, don't allow
redefinition or reextern. if function.name != self.name:
function.delete() function =
g\_llvm\_module.get\_function\_named(self.name) {% endhighlight %}

The Module symbol table works just like the Function symbol table when
it comes to name conflicts: if a new function is created with a name was
previously added to the symbol table, it will get implicitly renamed
when added to the Module. The code above exploits this fact to determine
if there was a previous definition of this function.

In Kaleidoscope, we choose to allow redefinitions of functions in two
cases: first, we want to allow 'extern'ing a function more than once, as
long as the prototypes for the externs match (since all arguments have
the same type, we just have to check that the number of arguments
match). Second, we want to allow 'extern'ing a function and then
defining a body for it. This is useful when defining mutually recursive
functions.

In order to implement this, the code above first checks to see if there
is a collision on the name of the function. If so, it deletes the
function we just created (by calling ``delete``) and then calling
``get_function_named`` to get the existing function with the specified
name.

{% highlight python %} # If the function already has a body, reject
this. if not function.is\_declaration: raise RuntimeError('Redefinition
of function.')

::

      # If F took a different number of args, reject.
      if len(callee.args) != len(self.args):
        raise RuntimeError('Redeclaration of a function with different number '
                           'of args.')

{% endhighlight %}

In order to verify the logic above, we first check to see if the
pre-existing function is a forward declaration. Since we don't allow
anything after a full definition of the function, the code rejects this
case. If the previous reference to a function was an 'extern', we simply
verify that the number of arguments for that definition and this one
match up. If not, we emit an error.

{% highlight python %} # Set names for all arguments and add them to the
variables symbol table. for arg, arg\_name in zip(function.args,
self.args): arg.name = arg\_name # Add arguments to variable symbol
table. g\_named\_values[arg\_name] = arg

::

    return function

{% endhighlight %}

The last bit of code for prototypes loops over all of the arguments in
the function, setting the name of the LLVM Argument objects to match,
and registering the arguments in the ``g_named_values`` map for future
use by the ``VariableExpressionNode``. Note that we don't check for
conflicting argument names here (e.g. "extern foo(a b a)"). Doing so
would be very straight-forward with the mechanics we have already used
above. Once this is all set up, it returns the Function object to the
caller.

{% highlight python %} def CodeGen(self): # Clear scope.
g\_named\_values.clear()

::

    # Create a function object.
    function = self.prototype.CodeGen()

{% endhighlight %}

Code generation for function definitions starts out simply enough: we
just clear out the ``g_named_values`` dictionary to make sure that there
isn't anything in it from the last function we compiled and codegen the
prototype. Code generation of the prototype ensures that there is an
LLVM Function object that is ready to go for us.

{% highlight python %} # Create a new basic block to start insertion
into. block = function.append\_basic\_block('entry') global
g\_llvm\_builder g\_llvm\_builder = Builder.new(block) {% endhighlight
%}

Now we get to the point where ``g_llvm_builder`` is set up. The first
line creates a new `basic
block <http://en.wikipedia.org/wiki/Basic_block>`_ (named "entry"),
which is inserted into the function. The second line declares that the
global ``g_llvm_builder`` object is to be changed. The last line creates
a new builder that is set up to insert new instructions into the basic
block we just created. Basic blocks in LLVM are an important part of
functions that define the `Control Flow
Graph <http://en.wikipedia.org/wiki/Control_flow_graph>`_. Since we
don't have any control flow, our functions will only contain one block
at this point. We'll fix this in `Chapter 5 <PythonLangImpl5.html>`_ :).

{% highlight python %} # Finish off the function. try: return\_value =
self.body.CodeGen() g\_llvm\_builder.ret(return\_value)

::

      # Validate the generated code, checking for consistency.
      function.verify()

{% endhighlight %}

Once the insertion point is set up, we call the ``CodeGen`` method for
the root expression of the function. If no error happens, this emits
code to compute the expression into the entry block and returns the
value that was computed. Assuming no error, we then create an LLVM `ret
instruction <http://www.llvm.org/docs/LangRef.html#i_ret>`_, which
completes the function. Once the function is built, we call ``verify``,
which is provided by LLVM. This function does a variety of consistency
checks on the generated code, to determine if our compiler is doing
everything right. Using this is important: it can catch a lot of bugs.
Once the function is finished and validated, we return it.

{% highlight python %} except: function.delete() raise

::

    return function

{% endhighlight %}

The only piece left here is handling of the error case. For simplicity,
we handle this by merely deleting the function we produced with the
``delete`` method. This allows the user to redefine a function that they
incorrectly typed in before: if we didn't delete it, it would live in
the symbol table, with a body, preventing future redefinition.

This code does have a bug, though. Since the ``PrototypeNode::CodeGen``
can return a previously defined forward declaration, our code can
actually delete a forward declaration. There are a number of ways to fix
this bug; see what you can come up with! Here is a testcase:

{% highlight python %} extern foo(a b) # ok, defines foo. def foo(a b) c
# error, 'c' is invalid. def bar() foo(1, 2) # error, unknown function
"foo" {% endhighlight %}

--------------

Driver Changes and Closing Thoughts # {#driver}
===============================================

For now, code generation to LLVM doesn't really get us much, except that
we can look at the pretty IR calls. The sample code inserts calls to
CodeGen into the ``Handle*`` functions, and then dumps out the LLVM IR.
This gives a nice way to look at the LLVM IR for simple functions. For
example:

{% highlight bash %} ready> 4+5 Read a top-level expression: define
double @0() { entry: ret double 9.000000e+00 } {% endhighlight %}

Note how the parser turns the top-level expression into anonymous
functions for us. This will be handy when we add JIT support in the next
chapter. Also note that the code is very literally transcribed, no
optimizations are being performed except simple constant folding done by
the Builder. We will add optimizations explicitly in the next chapter.

{% highlight bash %} ready> def foo(a b) a\ *a + 2*\ a\ *b + b*\ b Read
a function definition: define double @foo(double %a, double %b) { entry:
%multmp = fmul double %a, %a ; [#uses=1] %multmp1 = fmul double
2.000000e+00, %a ; [#uses=1] %multmp2 = fmul double %multmp1, %b ;
[#uses=1] %addtmp = fadd double %multmp, %multmp2 ; [#uses=1] %multmp3 =
fmul double %b, %b ; [#uses=1] %addtmp4 = fadd double %addtmp, %multmp3
; [#uses=1] ret double %addtmp4 } {% endhighlight %}

This shows some simple arithmetic. Notice the striking similarity to the
LLVM builder calls that we use to create the instructions.

{% highlight bash %} ready> def bar(a) foo(a, 4.0) + bar(31337) Read a
function definition: define double @bar(double %a) { entry: %calltmp =
call double @foo(double %a, double 4.000000e+00) ; [#uses=1] %calltmp1 =
call double @bar(double 3.133700e+04) ; [#uses=1] %addtmp = fadd double
%calltmp, %calltmp1 ; [#uses=1] ret double %addtmp } {% endhighlight %}

This shows some function calls. Note that this function will take a long
time to execute if you call it. In the future we'll add conditional
control flow to actually make recursion useful :).

{% highlight bash %} ready> extern cos(x) Read extern: declare double
@cos(double)

ready> cos(1.234) Read a top-level expression: define double @1() {
entry: %calltmp = call double @cos(double 1.234000e+00) ; [#uses=1] ret
double %calltmp } {% endhighlight %}

This shows an extern for the libm "cos" function, and a call to it.

{% highlight bash %} ready> ^C ; ModuleID = 'my cool jit'

define double @0() { entry: ret double 9.000000e+00 }

define double @foo(double %a, double %b) { entry: %multmp = fmul double
%a, %a ; [#uses=1] %multmp1 = fmul double 2.000000e+00, %a ; [#uses=1]
%multmp2 = fmul double %multmp1, %b ; [#uses=1] %addtmp = fadd double
%multmp, %multmp2 ; [#uses=1] %multmp3 = fmul double %b, %b ; [#uses=1]
%addtmp4 = fadd double %addtmp, %multmp3 ; [#uses=1] ret double %addtmp4
}

define double @bar(double %a) { entry: %calltmp = call double
@foo(double %a, double 4.000000e+00) ; [#uses=1] %calltmp1 = call double
@bar(double 3.133700e+04) ; [#uses=1] %addtmp = fadd double %calltmp,
%calltmp1 ; [#uses=1] ret double %addtmp }

declare double @cos(double)

define double @1() { entry: %calltmp = call double @cos(double
1.234000e+00) ; [#uses=1] ret double %calltmp } {% endhighlight %}

When you quit the current demo, it dumps out the IR for the entire
module generated. Here you can see the big picture with all the
functions referencing each other.

This wraps up the third chapter of the Kaleidoscope tutorial. Up next,
we'll describe how to `add JIT codegen and optimizer
support <PythonLangImpl4.html>`_ to this so we can actually start
running code!

--------------

Full Code Listing # {#code}
===========================

Here is the complete code listing for our running example, enhanced with
the LLVM code generator. Because this uses the llvmpy libraries, you
need to `download <../download.html>`_ and
`install <../userguide.html#install>`_ them.

{% highlight python %} #!/usr/bin/env python

import re from llvm.core import Module, Constant, Type, Function,
Builder, FCMP\_ULT

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

def HandleTopLevelExpression(self): self.Handle(self.ParseTopLevelExpr,
'Read a top-level expression:')

def Handle(self, function, message): try: print message,
function().CodeGen() except Exception, e: print 'Error:', e try:
self.Next() # Skip for error recovery. except: pass

Main driver code.
-----------------

def main(): # Install standard binary operators. # 1 is lowest possible
precedence. 40 is the highest. operator\_precedence = { '<': 10, '+':
20, '-': 20, '\*': 40 }

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

**`Next: Adding JIT and Optimizer Support <PythonLangImpl4.html>`_**
