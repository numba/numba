**********************************************************************
Chapter 6: Extending the Language: User-defined Operators
**********************************************************************

Written by `Chris Lattner <mailto:sabre@nondot.org>`_ and `Max
Shawabkeh <http://max99x.com>`_

Introduction # {#intro}
=======================

Welcome to Chapter 6 of the `Implementing a language with
LLVM <http://www.llvm.org/docs/tutorial/index.html>`_ tutorial. At this
point in our tutorial, we now have a fully functional language that is
fairly minimal, but also useful. There is still one big problem with it,
however. Our language doesn't have many useful operators (like division,
logical negation, or even any comparisons besides less-than).

This chapter of the tutorial takes a wild digression into adding
user-defined operators to the simple and beautiful Kaleidoscope
language. This digression now gives us a simple and ugly language in
some ways, but also a powerful one at the same time. One of the great
things about creating your own language is that you get to decide what
is good or bad. In this tutorial we'll assume that it is okay to use
this as a way to show some interesting parsing techniques.

At the end of this tutorial, we'll run through an example Kaleidoscope
application that `renders the Mandelbrot set <#example>`_. This gives an
example of what you can build with Kaleidoscope and its feature set.

User-defined Operators: the Idea # {#idea}
==========================================

The "operator overloading" that we will add to Kaleidoscope is more
general than languages like C++. In C++, you are only allowed to
redefine existing operators: you can't programatically change the
grammar, introduce new operators, change precedence levels, etc. In this
chapter, we will add this capability to Kaleidoscope, which will let the
user round out the set of operators that are supported.

The point of going into user-defined operators in a tutorial like this
is to show the power and flexibility of using a hand-written parser.
Thus far, the parser we have been implementing uses recursive descent
for most parts of the grammar and operator precedence parsing for the
expressions. See `Chapter 2 <PythonLangImpl2.html>`_ for details.
Without using operator precedence parsing, it would be very difficult to
allow the programmer to introduce new operators into the grammar: the
grammar is dynamically extensible as the JIT runs.

The two specific features we'll add are programmable unary operators
(right now, Kaleidoscope has no unary operators at all) as well as
binary operators. An example of this is:

{% highlight python %} # Logical unary not. def unary!(v) if v then 0
else 1

Define > with the same precedence as <.
=======================================

def binary> 10 (LHS RHS) RHS < LHS

Binary "logical or", (note that it does not "short circuit").
=============================================================

def binary\| 5 (LHS RHS) if LHS then 1 else if RHS then 1 else 0

Define = with slightly lower precedence than relationals.
=========================================================

def binary= 9 (LHS RHS) !(LHS < RHS \| LHS > RHS) {% endhighlight %}

Many languages aspire to being able to implement their standard runtime
library in the language itself. In Kaleidoscope, we can implement
significant parts of the language in the library!

We will break down implementation of these features into two parts:
implementing support for user-defined binary operators and adding unary
operators.

--------------

User-defined Binary Operators # {#binary}
=========================================

Adding support for user-defined binary operators is pretty simple with
our current framework. We'll first add support for the unary/binary
keywords:

{% highlight python %} class InToken(object): pass class
BinaryToken(object): pass class UnaryToken(object): pass ... def
Tokenize(string): ... elif identifier == 'in': yield InToken() elif
identifier == 'binary': yield BinaryToken() elif identifier == 'unary':
yield UnaryToken() else: yield IdentifierToken(identifier) {%
endhighlight %}

This just adds lexer support for the unary and binary keywords, like we
did in `previous chapters <PythonLangImpl5.html#iflexer>`_. One nice
thing about our current AST, is that we represent binary operators with
full generalisation by using their ASCII code as the opcode. For our
extended operators, we'll use this same representation, so we don't need
any new AST or parser support.

On the other hand, we have to be able to represent the definitions of
these new operators, in the "def binary\| 5" part of the function
definition. In our grammar so far, the "name" for the function
definition is parsed as the "prototype" production and into the
``PrototypeNode``. To represent our new user-defined operators as
prototypes, we have to extend the ``PrototypeNode`` like this:

{% highlight python %} # This class represents the "prototype" for a
function, which captures its name, # and its argument names (thus
implicitly the number of arguments the function # takes), as well as if
it is an operator. class PrototypeNode(object):

def **init**\ (self, name, args, is\_operator=False, precedence=0):
self.name = name self.args = args self.is\_operator = is\_operator
self.precedence = precedence

def IsBinaryOp(self): return self.is\_operator and len(self.args) == 2

def GetOperatorName(self): assert self.is\_operator return self.name[-1]

def CodeGen(self): ... {% endhighlight %}

Basically, in addition to knowing a name for the prototype, we now keep
track of whether it was an operator, and if it was, what precedence
level the operator is at. The precedence is only used for binary
operators (as you'll see below, it just doesn't apply for unary
operators). Now that we have a way to represent the prototype for a
user-defined operator, we need to parse it:

{% highlight python %} # prototype # ::= id '(' id\* ')' # ::= binary
LETTER number? (id, id) # ::= unary LETTER (id) def
ParsePrototype(self): precedence = None if isinstance(self.current,
IdentifierToken): kind = 'normal' function\_name = self.current.name
self.Next() # eat function name. elif isinstance(self.current,
BinaryToken): kind = 'binary' self.Next() # eat 'binary'. if not
isinstance(self.current, CharacterToken): raise RuntimeError('Expected
an operator after "binary".') function\_name = 'binary' +
self.current.char self.Next() # eat the operator. if
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

    if kind == 'binary' and len(arg_names) != 2:
      raise RuntimeError('Invalid number of arguments for a binary operator.')

    return PrototypeNode(function_name, arg_names, kind != 'normal', precedence)

{% endhighlight %}

This is all fairly straightforward parsing code, and we have already
seen a lot of similar code in the past. One interesting part about the
code above is the couple lines that set up ``function_name`` for
operators. This builds names like "binary@" for a newly defined "@"
operator. This then takes advantage of the fact that symbol names in the
LLVM symbol table are allowed to have any character in them.

The next interesting thing to add, is codegen support for these binary
operators. Given our current structure, this is a simple addition of a
default case for our existing binary operator node:

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
      function = g_llvm_module.get_function_named('binary' + self.operator)
      return g_llvm_builder.call(function, [left, right], 'binop')

{% endhighlight %}

As you can see above, the new code is actually really simple. It just
does a lookup for the appropriate operator in the symbol table and
generates a function call to it. Since user-defined operators are just
built as normal functions (because the "prototype" boils down to a
function with the right name) everything falls into place.

The final piece of code we are missing, is a bit of top-level magic. We
will need to make the dinary precedence map global and modify it
whenever we define a new binary operator:

{% highlight python %} # The binary operator precedence chart.
g\_binop\_precedence = {} ... class FunctionNode(object): ... def
CodeGen(self): ... # Create a function object. function =
self.prototype.CodeGen()

::

    # If this is a binary operator, install its precedence.
    if self.prototype.IsBinaryOp():
      operator = self.prototype.GetOperatorName()
      g_binop_precedence[operator] = self.prototype.precedence
    ...
    # Finish off the function.
    try:
      ...
    except:
      function.delete()
      if self.prototype.IsBinaryOp():
        del g_binop_precedence[self.prototype.GetOperatorName()]
      raise

    return function

... def main(): ... g\_binop\_precedence['<'] = 10
g\_binop\_precedence['+'] = 20 g\_binop\_precedence['-'] = 20
g\_binop\_precedence['\*'] = 40 ... {% endhighlight %}

Basically, before CodeGening a function, if it is a user-defined
operator, we register it in the precedence table. This allows the binary
operator parsing logic we already have in place to handle it. Since we
are working on a fully-general operator precedence parser, this is all
we need to do to "extend the grammar".

Now we have useful user-defined binary operators. This builds a lot on
the previous framework we built for other operators. Adding unary
operators is a bit more challenging, because we don't have any framework
for it yet - let's see what it takes.

User-defined Unary Operators # {#unary}
=======================================

Since we don't currently support unary operators in the Kaleidoscope
language, we'll need to add everything to support them. Above, we added
simple support for the 'unary' keyword to the lexer. In addition to
that, we need an AST node:

{% highlight python %} # Expression class for a unary operator. class
UnaryExpressionNode(ExpressionNode):

def **init**\ (self, operator, operand): self.operator = operator
self.operand = operand

def CodeGen(self): ... {% endhighlight %}

This AST node is very simple and obvious by now. It directly mirrors the
binary operator AST node, except that it only has one child. With this,
we need to add the parsing logic. Parsing a unary operator is pretty
simple: we'll add a new function to do it:

{% highlight python %} # unary ::= primary \| unary\_operator unary def
ParseUnary(self): # If the current token is not an operator, it must be
a primary expression. if (not isinstance(self.current, CharacterToken)
or self.current in [CharacterToken('('), CharacterToken(',')]): return
self.ParsePrimary()

::

    # If this is a unary operator, read it.
    operator = self.current.char
    self.Next()  # eat the operator.
    return UnaryExpressionNode(operator, self.ParseUnary())

{% endhighlight %}

The grammar we add is pretty straightforward here. If we see a unary
operator when parsing a primary operator, we eat the operator as a
prefix and parse the remaining piece as another unary operator. This
allows us to handle multiple unary operators (e.g. ``!!x``). Note that
unary operators can't have ambiguous parses like binary operators can,
so there is no need for precedence information.

The problem with this function, is that we need to call ParseUnary from
somewhere. To do this, we change previous callers of ParsePrimary to
call ParseUnary instead:

{% highlight python %} # binoprhs ::= (binary\_operator unary)\* def
ParseBinOpRHS(self, left, left\_precedence): ... # Parse the unary
expression after the binary operator. right = self.ParseUnary() ...

# expression ::= unary binoprhs def ParseExpression(self): left =
self.ParseUnary() return self.ParseBinOpRHS(left, 0) {% endhighlight %}

With these two simple changes, we are now able to parse unary operators
and build the AST for them. Next up, we need to add parser support for
prototypes, to parse the unary operator prototype. We extend the binary
operator code above with:

{% highlight python %} # prototype # ::= id '(' id\* ')' # ::= binary
LETTER number? (id, id) # ::= unary LETTER (id) def
ParsePrototype(self): precedence = None if isinstance(self.current,
IdentifierToken): ... elif isinstance(self.current, UnaryToken): kind =
'unary' self.Next() # eat 'unary'. if not isinstance(self.current,
CharacterToken): raise RuntimeError('Expected an operator after
"unary".') function\_name = 'unary' + self.current.char self.Next() #
eat the operator. elif isinstance(self.current, BinaryToken): ... else:
raise RuntimeError('Expected function name, "unary" or "binary" in '
'prototype.') ... if kind == 'unary' and len(arg\_names) != 1: raise
RuntimeError('Invalid number of arguments for a unary operator.') elif
kind == 'binary' and len(arg\_names) != 2: raise RuntimeError('Invalid
number of arguments for a binary operator.')

::

    return PrototypeNode(function_name, arg_names, kind != 'normal', precedence)

{% endhighlight %}

As with binary operators, we name unary operators with a name that
includes the operator character. This assists us at code generation
time. Speaking of, the final piece we need to add is codegen support for
unary operators. It looks like this:

{% highlight python %} class UnaryExpressionNode(ExpressionNode): ...
def CodeGen(self): operand = self.operand.CodeGen() function =
g\_llvm\_module.get\_function\_named('unary' + self.operator) return
g\_llvm\_builder.call(function, [operand], 'unop') {% endhighlight %}

This code is similar to, but simpler than, the code for binary
operators. It is simpler primarily because it doesn't need to handle any
predefined operators.

--------------

Kicking the Tires # {#example}
==============================

It is somewhat hard to believe, but with a few simple extensions we've
covered in the last chapters, we have grown a real-ish language. With
this, we can do a lot of interesting things, including I/O, math, and a
bunch of other things. For example, we can now add a nice sequencing
operator (assuming we import ``putchard`` as described in Chapter 4):

{% highlight python %} ready> def binary : 1 (x y) 0 # Low-precedence
operator that ignores operands. ... ready> extern putchard(x) ... ready>
def printd(x) putchard(x) : putchard(10) .. ready> printd(65) :
printd(66) : printd(67) A B C Evaluated to: 0.0 {% endhighlight %}

We can also define a bunch of other "primitive" operations, such as:

{% highlight python %} # Logical unary not. def unary!(v) if v then 0
else 1

Unary negate.
=============

def unary-(v) 0-v

Define > with the same precedence as <.
=======================================

def binary> 10 (LHS RHS) RHS < LHS

Binary logical or, which does not short circuit.
================================================

def binary\| 5 (LHS RHS) if LHS then 1 else if RHS then 1 else 0

Binary logical and, which does not short circuit.
=================================================

def binary& 6 (LHS RHS) if !LHS then 0 else !!RHS

Define = with slightly lower precedence than relationals.
=========================================================

def binary = 9 (LHS RHS) !(LHS < RHS \| LHS > RHS)

{% endhighlight %}

Given the previous if/then/else support, we can also define interesting
functions for I/O. For example, the following prints out a character
whose "density" reflects the value passed in: the lower the value, the
denser the character:

{% highlight python %} ready>

extern putchard(char) def printdensity(d) if d > 8 then putchard(32) # '
' else if d > 4 then putchard(46) # '.' else if d > 2 then putchard(43)
# '+' else putchard(42); # '*' ... ready> printdensity(1):
printdensity(2): printdensity(3) : printdensity(4): printdensity(5):
printdensity(9): putchard(10)*\ ++.. Evaluated to 0.000000 {%
endhighlight %}

Based on these simple primitive operations, we can start to define more
interesting things. For example, here's a little function that solves
for the number of iterations it takes a function in the complex plane to
converge:

{% highlight python %} # determine whether the specific location
diverges. # Solve for z = z^2 + c in the complex plane. def
mandelconverger(real imag iters creal cimag) if iters > 255 \|
(real\ *real + imag*\ imag > 4) then iters else
mandelconverger(real\ *real - imag*\ imag + creal, 2\ *real*\ imag +
cimag, iters+1, creal, cimag)

return the number of iterations required for the iteration to escape
====================================================================

def mandelconverge(real imag) mandelconverger(real, imag, 0, real, imag)
{% endhighlight %}

This "z = z2 + c" function is a beautiful little creature that is the
basis for computation of the `Mandelbrot
Set <http://en.wikipedia.org/wiki/Mandelbrot_set>`_. Our
``mandelconverge`` function returns the number of iterations that it
takes for a complex orbit to escape, saturating to 255. This is not a
very useful function by itself, but if you plot its value over a
two-dimensional plane, you can see the Mandelbrot set. Given that we are
limited to using putchard here, our amazing graphical output is limited,
but we can whip together something using the density plotter above:

{% highlight python %} # compute and plot the mandlebrot set with the
specified 2 dimensional range # info. def mandelhelp(xmin xmax xstep
ymin ymax ystep) for y = ymin, y < ymax, ystep in ( (for x = xmin, x <
xmax, xstep in printdensity(mandleconverge(x,y))) : putchard(10) )

mandel - This is a convenient helper function for ploting the mandelbrot set
============================================================================

from the specified position with the specified Magnification.
=============================================================

def mandel(realstart imagstart realmag imagmag) mandelhelp(realstart,
realstart+realmag\ *78, realmag, imagstart, imagstart+imagmag*\ 40,
imagmag); {% endhighlight %}

Given this, we can try plotting out the mandlebrot set! Lets try it out:

{% highlight bash %} ready> mandel(-2.3, -1.3, 0.05, 0.07)
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++...++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++..
...+++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++..
..+++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++.
..++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++....
..++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++.......
.....++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++. . ...
.++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++...
++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++....
.+++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++..+++++....
..+++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++. ..........
+++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*++++++++.. ..
.++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*++++++++++...
.++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*\*\*\*\*++++++++++..
.++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*\*\*\*++++++.....
..++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*\*\*\*+........
...++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*\*\*\*+... ....
...++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*\*\*\*+++++......
..++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*\*\*\*++++++++++...
.++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*++++++++++...
++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*+++++++++.. ..
..++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*++++++.. ..........
+++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++...+++.....
..+++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++....
..++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++...
+++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++.. . ...
.++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++.......
......+++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++....
..++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++..
..++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++..
...+++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++..
...+++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++....+++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
Evaluated to 0.0 ready> mandel(-2, -1, 0.02, 0.04)
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++++++
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++++...
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++++.......
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++..........
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++...
...
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++......
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++.......
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++++..........
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++++...........
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++++++.........
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++...........+++++..............
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++....
.........................
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++.... .........
............ \*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++++..... ......
\*\*\*\*\*\*\*\*+++++++++++++++++++++++.......
\*\*\*\*\*\*+++++++++++++++++++++++++........
\*\*\*\*+++++++++++++++++++++++++.......
***+++++++++++++++++++++++.........**\ ++++++++++++++++...........*\ ++++++++++++................
\*++++....................

*++++....................*\ ++++++++++++................
**++++++++++++++++...........**\ *+++++++++++++++++++++++.........
\*\*\*\*+++++++++++++++++++++++++.......
\*\*\*\*\*\*+++++++++++++++++++++++++........
\*\*\*\*\*\*\*\*+++++++++++++++++++++++.......
\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++++..... ......
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++.... .........
............ \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++....
.........................
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++...........+++++..............
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++++++.........
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++++...........
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++++..........
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++.......
Evaluated to: 0.0 ready> mandel(-0.9, -1.4, 0.02, 0.03)
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++...++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++.. .
.++++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++...
......++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*\*\*+++++++++++++++++++...
.......+++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++.... ....
..++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*\*\*++++++++++++++++++++++......
...++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*\*\*+++++++++++++++++++++++.......
.....++++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*\*\*++++++++++++++++++++++++.......
.....+++++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
\*\*\*\*+++++++++++++++++++++++++.... .
.....+++++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*
**+++++++++++++++++++++++++....
...++++++++++++++++**\ \*\*\*\*\*\*\*\*\*\*\**\ +++++++++++++++++++++++.......
....++++++++++++++++\*\*\*\*\*\*\*\*\*\*\*\*
+++++++++++++++++++++..........
.....++++++++++++++++\*\*\*\*\*\*\*\*\*\*\*
++++++++++++++++++.............
.......+++++++++++++++\*\*\*\*\*\*\*\*\*\*
+++++++++++++++................
............++++++++++\*\*\*\*\*\*\*\*\*\*
+++++++++++++................. .................+++++\*\*\*\*\*\*\*\*\*
+++++++++++... .... .......... .+++++\*\*\*\*\*\*\*\* ++++++++++.....
........ ...+++++\*\*\*\*\*\*\* ++++++++...... ..++++++\*\*\*\*\*\*
+++++++........ ..+++++\*\*\*\*\*\* +++++.......... ..++++++\*\*\*\*\*
++++.......... ....++++++\*\*\*\*\* ++.......... ....+++++++\*\*\*\*
.......... ......+++++++\ **\* .......... .....+++++++**\ \* ..........
.....++++++\ **\* ......... .+++++++** ........ .+++++++\ *\* ......
...+++++++* . ....++++++++\* ...++++++++\* ..+++++++++ ..+++++++++
Evaluated to: 0.0 ready> ^C {% endhighlight %}

At this point, you may be starting to realize that Kaleidoscope is a
real and powerful language. It may not be self-similar :), but it can be
used to plot things that are!

With this, we conclude the "adding user-defined operators" chapter of
the tutorial. We have successfully augmented our language, adding the
ability to extend the language in the library, and we have shown how
this can be used to build a simple but interesting end-user application
in Kaleidoscope. At this point, Kaleidoscope can build a variety of
applications that are functional and can call functions with
side-effects, but it can't actually define and mutate a variable itself.

Strikingly, variable mutation is an important feature of some languages,
and it is not at all obvious how to `add support for mutable
variables <PythonLangImpl7.html>`_ without having to add an "SSA
construction" phase to your front-end. In the next chapter, we will
describe how you can add variable mutation without building SSA in your
front-end.

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

The binary operator precedence chart.
=====================================

g\_binop\_precedence = {}

Lexer
-----

The lexer yields one of these types for each token.
===================================================

class EOFToken(object): pass class DefToken(object): pass class
ExternToken(object): pass class IfToken(object): pass class
ThenToken(object): pass class ElseToken(object): pass class
ForToken(object): pass class InToken(object): pass class
BinaryToken(object): pass class UnaryToken(object): pass

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

Expression class for a unary operator.
======================================

class UnaryExpressionNode(ExpressionNode):

def **init**\ (self, operator, operand): self.operator = operator
self.operand = operand

def CodeGen(self): operand = self.operand.CodeGen() function =
g\_llvm\_module.get\_function\_named('unary' + self.operator) return
g\_llvm\_builder.call(function, [operand], 'unop')

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

    # If this is a binary operator, install its precedence.
    if self.prototype.IsBinaryOp():
      operator = self.prototype.GetOperatorName()
      g_binop_precedence[operator] = self.prototype.precedence

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

# primary ::= identifierexpr \| numberexpr \| parenexpr \| ifexpr \|
forexpr def ParsePrimary(self): if isinstance(self.current,
IdentifierToken): return self.ParseIdentifierExpr() elif
isinstance(self.current, NumberToken): return self.ParseNumberExpr()
elif isinstance(self.current, IfToken): return self.ParseIfExpr() elif
isinstance(self.current, ForToken): return self.ParseForExpr() elif
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
40 is the highest. g\_binop\_precedence['<'] = 10
g\_binop\_precedence['+'] = 20 g\_binop\_precedence['-'] = 20
g\_binop\_precedence['\*'] = 40

# Run the main "interpreter loop". while True: print 'ready>', try: raw
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

**`Next: Extending the language: mutable variables / SSA
construction <PythonLangImpl7.html>`_**
