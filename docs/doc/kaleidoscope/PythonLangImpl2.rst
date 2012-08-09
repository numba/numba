***************************************************
Chapter 2: Implementing a Parser and AST
***************************************************

Written by `Chris Lattner <mailto:sabre@nondot.org>`_ and `Max
Shawabkeh <http://max99x.com>`_

Introduction # {#intro}
=======================

Welcome to Chapter 2 of the `Implementing a language with
LLVM <http://www.llvm.org/docs/tutorial/index.html>`_ tutorial. This
chapter shows you how to use the lexer, built in `Chapter
1 <PythonLangImpl1.html>`_, to build a full
`parser <http://en.wikipedia.org/wiki/Parsing>`_ for our Kaleidoscope
language. Once we have a parser, we'll define and build an `Abstract
Syntax Tree <http://en.wikipedia.org/wiki/Abstract_syntax_tree>`_ (AST).

The parser we will build uses a combination of `Recursive Descent
Parsing <http://en.wikipedia.org/wiki/Recursive_descent_parser>`_ and
`Operator-Precedence
Parsing <http://en.wikipedia.org/wiki/Operator-precedence_parser>`_ to
parse the Kaleidoscope language (the latter for binary expressions and
the former for everything else). Before we get to parsing though, lets
talk about the output of the parser: the Abstract Syntax Tree.

--------------

The Abstract Syntax Tree (AST) # {#ast}
=======================================

The AST for a program captures its behavior in such a way that it is
easy for later stages of the compiler (e.g. code generation) to
interpret. We basically want one object for each construct in the
language, and the AST should closely model the language. In
Kaleidoscope, we have expressions, a prototype, and a function object.
We'll start with expressions first:

{% highlight python %} # Base class for all expression nodes. class
ExpressionNode(object): pass

Expression class for numeric literals like "1.0".
=================================================

class NumberExpressionNode(ExpressionNode): def **init**\ (self, value):
self.value = value

{% endhighlight %}

The code above shows the definition of the base ExpressionNode class and
one subclass which we use for numeric literals. The important thing to
note about this code is that the NumberExpressionNode class captures the
numeric value of the literal as an instance variable. This allows later
phases of the compiler to know what the stored numeric value is.

Right now we only create the AST, so there are no useful methods on
them. It would be very easy to add a virtual method to pretty print the
code, for example. Here are the other expression AST node definitions
that we'll use in the basic form of the Kaleidoscope language:

{% highlight python %} # Expression class for referencing a variable,
like "a". class VariableExpressionNode(ExpressionNode): def
**init**\ (self, name): self.name = name

Expression class for a binary operator.
=======================================

class BinaryOperatorExpressionNode(ExpressionNode): def **init**\ (self,
operator, left, right): self.operator = operator self.left = left
self.right = right

Expression class for function calls.
====================================

class CallExpressionNode(ExpressionNode): def **init**\ (self, callee,
args): self.callee = callee self.args = args {% endhighlight %}

This is all (intentionally) rather straight-forward: variables capture
the variable name, binary operators capture their opcode (e.g. '+'), and
calls capture a function name as well as a list of any argument
expressions. One thing that is nice about our AST is that it captures
the language features without talking about the syntax of the language.
Note that there is no discussion about precedence of binary operators,
lexical structure, etc.

For our basic language, these are all of the expression nodes we'll
define. Because it doesn't have conditional control flow, it isn't
Turing-complete; we'll fix that in a later installment. The two things
we need next are a way to talk about the interface to a function, and a
way to talk about functions themselves:

{% highlight python %} # This class represents the "prototype" for a
function, which captures its name, # and its argument names (thus
implicitly the number of arguments the function # takes). class
PrototypeNode(object): def **init**\ (self, name, args): self.name =
name self.args = args

This class represents a function definition itself.
===================================================

class FunctionNode(object): def **init**\ (self, prototype, body):
self.prototype = prototype self.body = body {% endhighlight %}

In Kaleidoscope, functions are typed with just a count of their
arguments. Since all values are double precision floating point, the
type of each argument doesn't need to be stored anywhere. In a more
aggressive and realistic language, the ``ExpressionNode`` class would
probably have a type field.

With this scaffolding, we can now talk about parsing expressions and
function bodies in Kaleidoscope.

--------------

Parser Basics # {#parserbasics}
===============================

Now that we have an AST to build, we need to define the parser code to
build it. The idea here is that we want to parse something like
``x + y`` (which is returned as three tokens by the lexer) into an AST
that could be generated with calls like this:

{% highlight python %} x = VariableExpressionNode('x') y =
VariableExpressionNode('y') result = BinaryOperatorExpressionNode('+',
x, y) {% endhighlight %}

In order to do this, we'll start by defining a lightweight ``Parser``
class with some basic helper routines:

{% highlight python %} class Parser(object):

def **init**\ (self, tokens, binop\_precedence): self.tokens = tokens
self.binop\_precedence = binop\_precedence self.Next()

# Provide a simple token buffer. Parser.current is the current token the
# parser is looking at. Parser.Next() reads another token from the lexer
and # updates Parser.current with its results. def Next(self):
self.current = self.tokens.next() {% endhighlight %}

This implements a simple token buffer around the lexer. This allows us
to look one token ahead at what the lexer is returning. Every function
in our parser will assume that ``self.current`` is the current token
that needs to be parsed. Note that the first token is read as soon as
the parser is instantiated. Let us ignore the ``binop_precedence``
parameter for now. It will be explained when we start `parsing binary
operators <#parserbinops>`_.

With these basic helper functions, we can implement the first piece of
our grammar: numeric literals.

--------------

Basic Expression Parsing # {#parserprimexprs}
=============================================

We start with numeric literals, because they are the simplest to
process. For each production in our grammar, we'll define a function
which parses that production. For numeric literals, we have:

{% highlight python %} # numberexpr ::= number def
ParseNumberExpr(self): result = NumberExpressionNode(self.current.value)
self.Next() # consume the number. return result {% endhighlight %}

This method is very simple: it expects to be called when the current
token is a ``NumberToken``. It takes the current number value, creates a
``NumberExpressionNode``, advances to the next token, and finally
returns.

There are some interesting aspects to this. The most important one is
that this routine eats all of the tokens that correspond to the
production and returns the lexer buffer with the next token (which is
not part of the grammar production) ready to go. This is a fairly
standard way to go for recursive descent parsers. For a better example,
the parenthesis operator is defined like this:

{% highlight python %} # parenexpr ::= '(' expression ')' def
ParseParenExpr(self): self.Next() # eat '('.

::

    contents = self.ParseExpression()

    if self.current != CharacterToken(')'):
      raise RuntimeError('Expected ")".')
    self.Next()  # eat ')'.

    return contents

{% endhighlight %}

This function illustrates an interesting aspect of the parser. The
function uses recursion by calling ``ParseExpression`` (we will soon see
that ``ParseExpression`` can call ``ParseParenExpr``). This is powerful
because it allows us to handle recursive grammars, and keeps each
production very simple. Note that parentheses do not cause construction
of AST nodes themselves. While we could do it this way, the most
important role of parentheses are to guide the parser and provide
grouping. Once the parser constructs the AST, parentheses are not
needed.

The next simple production is for handling variable references and
function calls:

{% highlight python %} # identifierexpr ::= identifier \| identifier '('
expression\* ')' def ParseIdentifierExpr(self): identifier\_name =
self.current.name self.Next() # eat identifier.

::

    if self.current != CharacterToken('('):  # Simple variable reference.
      return VariableExpressionNode(identifier_name);

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

{% endhighlight %}

This routine follows the same style as the other routines. It expects to
be called if the current token is an ``IdentifierToken``. It also has
recursion and error handling. One interesting aspect of this is that it
uses *look-ahead* to determine if the current identifier is a stand
alone variable reference or if it is a function call expression. It
handles this by checking to see if the token after the identifier is a
'(' token, constructing either a ``VariableExpressionNode`` or
``CallExpressionNode`` as appropriate.

Now that we have all of our simple expression-parsing logic in place, we
can define a helper function to wrap it together into one entry point.
We call this class of expressions "primary" expressions, for reasons
that will become more clear `later in the
tutorial <PythonLangImpl6.html#unary>`_. In order to parse an arbitrary
primary expression, we need to determine what sort of expression it is:

{% highlight python %} # primary ::= identifierexpr \| numberexpr \|
parenexpr def ParsePrimary(self): if isinstance(self.current,
IdentifierToken): return self.ParseIdentifierExpr() elif
isinstance(self.current, NumberToken): return self.ParseNumberExpr();
elif self.current == CharacterToken('('): return self.ParseParenExpr()
else: raise RuntimeError('Unknown token when expecting an expression.')
{% endhighlight %}

Now that you see the definition of this function, it is more obvious why
we can assume the state of ``Parser.current`` in the various functions.
This uses look-ahead to determine which sort of expression is being
inspected, and then parses it with a function call.

Now that basic expressions are handled, we need to handle binary
expressions. They are a bit more complex.

--------------

Binary Expression Parsing # {#parserbinops}
===========================================

Binary expressions are significantly harder to parse because they are
often ambiguous. For example, when given the string ``x+y*z``, the
parser can choose to parse it as either ``(x+y)*z`` or ``x+(y*z)``. With
common definitions from mathematics, we expect the later parse, because
``*`` (multiplication) has higher *precedence* than ``+`` (addition).

There are many ways to handle this, but an elegant and efficient way is
to use `Operator-Precedence
Parsing <http://en.wikipedia.org/wiki/Operator-precedence_parser>`_.
This parsing technique uses the precedence of binary operators to guide
recursion. To start with, we need a table of precedences. Remember the
``binop_precedence`` parameter we passed to the ``Parser`` constructor?
Now is the time to use it:

{% highlight python %} def main(): # Install standard binary operators.
# 1 is lowest possible precedence. 40 is the highest.
operator\_precedence = { '<': 10, '+': 20, '-': 20, '\*': 40 }

# Run the main ``interpreter loop``. while True:

::

    ...

    parser = Parser(Tokenize(raw), operator_precedence)

{% endhighlight %}

For the basic form of Kaleidoscope, we will only support 4 binary
operators (this can obviously be extended by you, our brave and intrepid
reader). Having a dictionary makes it easy to add new operators and
makes it clear that the algorithm doesn't depend on the specific
operators involved, but it would be easy enough to eliminate the map and
hardcode the comparisons.

We also define a helper function to get the precedence of the current
token, or -1 if the token is not a binary operator:

{% highlight python %} # Gets the precedence of the current token, or -1
if the token is not a binary # operator. def
GetCurrentTokenPrecedence(self): if isinstance(self.current,
CharacterToken): return self.binop\_precedence.get(self.current.char,
-1) else: return -1 {% endhighlight %}

With the helper above defined, we can now start parsing binary
expressions. The basic idea of operator precedence parsing is to break
down an expression with potentially ambiguous binary operators into
pieces. Consider, for example, the expression ``a+b+(c+d)*e*f+g``.
Operator precedence parsing considers this as a stream of primary
expressions separated by binary operators. As such, it will first parse
the leading primary expression ``a``, then it will see the pairs
``[+, b] [+, (c+d)] [*, e] [*, f] and [+, g]``. Note that because
parentheses are primary expressions, the binary expression parser
doesn't need to worry about nested subexpressions like (c+d) at all.

To start, an expression is a primary expression potentially followed by
a sequence of ``[binop,primaryexpr]`` pairs:

{% highlight python %} # expression ::= primary binoprhs def
ParseExpression(self): left = self.ParsePrimary() return
self.ParseBinOpRHS(left, 0) {% endhighlight %}

``ParseBinOpRHS`` is the function that parses the sequence of pairs for
us. It takes a precedence and a pointer to an expression for the part
that has been parsed so far. Note that ``x`` is a perfectly valid
expression: As such, ``binoprhs`` is allowed to be empty, in which case
it returns the expression that is passed into it. In our example above,
the code passes the expression for ``a`` into ``ParseBinOpRHS`` and the
current token is ``+``.

The precedence value passed into ``ParseBinOpRHS`` indicates the \*
minimal operator precedence\* that the function is allowed to eat. For
example, if the current pair stream is ``[+, x]`` and ``ParseBinOpRHS``
is passed in a precedence of 40, it will not consume any tokens (because
the precedence of '+' is only 20). With this in mind, ``ParseBinOpRHS``
starts with:

{% highlight python %} # binoprhs ::= (operator primary)\* def
ParseBinOpRHS(self, left, left\_precedence): # If this is a binary
operator, find its precedence. while True: precedence =
self.GetCurrentTokenPrecedence()

::

      # If this is a binary operator that binds at least as tightly as the
      # current one, consume it; otherwise we are done.
      if precedence < left_precedence:
        return left

{% endhighlight %}

This code gets the precedence of the current token and checks to see if
if is too low. Because we defined invalid tokens to have a precedence of
-1, this check implicitly knows that the pair-stream ends when the token
stream runs out of binary operators. If this check succeeds, we know
that the token is a binary operator and that it will be included in this
expression:

{% highlight python %} binary\_operator = self.current.char self.Next()
# eat the operator.

::

      # Parse the primary expression after the binary operator.
      right = self.ParsePrimary()

{% endhighlight %}

As such, this code eats (and remembers) the binary operator and then
parses the primary expression that follows. This builds up the whole
pair, the first of which is ``[+, b]`` for the running example.

Now that we parsed the left-hand side of an expression and one pair of
the RHS sequence, we have to decide which way the expression associates.
In particular, we could have ``(a+b) binop unparsed`` or
``a + (b binop unparsed)``. To determine this, we look ahead at
``binop`` to determine its precedence and compare it to BinOp's
precedence (which is '+' in this case):

{% highlight python %} # If binary\_operator binds less tightly with
right than the operator after # right, let the pending operator take
right as its left. next\_precedence = self.GetCurrentTokenPrecedence()
if precedence < next\_precedence: {% endhighlight %}

If the precedence of the binop to the right of ``RHS`` is lower or equal
to the precedence of our current operator, then we know that the
parentheses associate as ``(a+b) binop ...``. In our example, the
current operator is ``+`` and the next operator is ``+``, we know that
they have the same precedence. In this case we'll create the AST node
for ``a+b``, and then continue parsing:

{% highlight python %} if precedence < next\_precedence: ... if body
omitted ...

::

      # Merge left/right.
      left = BinaryOperatorExpressionNode(binary_operator, left, right);

{% endhighlight %}

In our example above, this will turn ``a+b+`` into ``(a+b)`` and execute
the next iteration of the loop, with ``+`` as the current token. The
code above will eat, remember, and parse ``(c+d)`` as the primary
expression, which makes the current pair equal to ``[+, (c+d)]``. It
will then evaluate the 'if' conditional above with ``*`` as the binop to
the right of the primary. In this case, the precedence of ``*`` is
higher than the precedence of ``+`` so the if condition will be entered.

The critical question left here is
``how can the if condition parse the right hand side in full``? In
particular, to build the AST correctly for our example, it needs to get
all of ``( c + d ) * e * f`` as the RHS expression variable. The code to
do this is surprisingly simple (code from the above two blocks
duplicated for context):

{% highlight python %} # If binary\_operator binds less tightly with
right than the operator after # right, let the pending operator take
right as its left. next\_precedence = self.GetCurrentTokenPrecedence()
if precedence < next\_precedence: right = self.ParseBinOpRHS(right,
precedence + 1)

::

      # Merge left/right.
      left = BinaryOperatorExpressionNode(binary_operator, left, right)

{% endhighlight %}

At this point, we know that the binary operator to the RHS of our
primary has higher precedence than the binop we are currently parsing.
As such, we know that any sequence of pairs whose operators are all
higher precedence than ``+`` should be parsed together and returned as
``RHS``. To do this, we recursively invoke the ``ParseBinOpRHS``
function specifying ``precedence + 1`` as the minimum precedence
required for it to continue. In our example above, this will cause it to
return the AST node for ``(c+d)*e*f`` as RHS, which is then set as the
RHS of the '+' expression.

Finally, on the next iteration of the while loop, the ``+g`` piece is
parsed and added to the AST. With this little bit of code (11
non-trivial lines), we correctly handle fully general binary expression
parsing in a very elegant way. This was a whirlwind tour of this code,
and it is somewhat subtle. I recommend running through it with a few
tough examples to see how it works.

This wraps up handling of expressions. At this point, we can point the
parser at an arbitrary token stream and build an expression from it,
stopping at the first token that is not part of the expression. Next up
we need to handle function definitions, etc.

--------------

Parsing the Rest # {#parsertop}
===============================

The next thing missing is handling of function prototypes. In
Kaleidoscope, these are used both for 'extern' function declarations as
well as function body definitions. The code to do this is
straight-forward and not very interesting (once you've survived
expressions):

{% highlight python %} # prototype ::= id '(' id\* ')' def
ParsePrototype(self): if not isinstance(self.current, IdentifierToken):
raise RuntimeError('Expected function name in prototype.')

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

{% endhighlight %}

Given this, a function definition is very simple, just a prototype plus
an expression to implement the body:

{% highlight python %} # definition ::= 'def' prototype expression def
ParseDefinition(self): self.Next() # eat def. proto =
self.ParsePrototype() body = self.ParseExpression() return
FunctionNode(proto, body) {% endhighlight %}

In addition, we support 'extern' to declare functions like 'sin' and
'cos' as well as to support forward declaration of user functions. These
'extern's are just prototypes with no body:

{% highlight python %} # external ::= 'extern' prototype def
ParseExtern(self): self.Next() # eat extern. return
self.ParsePrototype() {% endhighlight %}

Finally, we'll also let the user type in arbitrary top-level expressions
and evaluate them on the fly. We will handle this by defining anonymous
nullary (zero argument) functions for them:

{% highlight python %} # toplevelexpr ::= expression def
ParseTopLevelExpr(self): proto = PrototypeNode('', []) return
FunctionNode(proto, self.ParseExpression()) {% endhighlight %}

Now that we have all the pieces, let's build a little driver that will
let us actually *execute* this code we've built!

--------------

The Driver # {#driver}
======================

The driver for this simply invokes all of the parsing pieces with a
top-level dispatch loop. There isn't much interesting here, so I'll just
include the top-level loop. See `below <#code>`_ for full code.

{% highlight python %} # Run the main "interpreter loop". while True:
print 'ready>', try: raw = raw\_input() except KeyboardInterrupt: return

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

{% endhighlight %}

Here we create a new ``Parser`` for each line read, and try to parse out
all the expressions, declarations and definitions in the line. We also
allow the user to quit using Ctrl+C.

--------------

Conclusions # {#conclusions}
============================

With just under 330 lines of commented code (200 lines of non-comment,
non-blank code), we fully defined our minimal language, including a
lexer, parser, and AST builder. With this done, the executable will
validate Kaleidoscope code and tell us if it is grammatically invalid.
For example, here is a sample interaction:

{% highlight python %} $ python kaleidoscope.py ready> def foo(x y)
x+foo(y, 4.0) Parsed a function definition. ready> def foo(x y) x+y y
Parsed a function definition. Parsed a top-level expression. ready> def
foo(x y) x+y ) Parsed a function definition. Error: Unknown token when
expecting an expression. ready> extern sin(a); Parsed an extern. ready>
^C $ {% endhighlight %}

There is a lot of room for extension here. You can define new AST nodes,
extend the language in many ways, etc. In the `next
installment <PythonLangImpl3.html>`_, we will describe how to generate
LLVM Intermediate Representation (IR) from the AST.

--------------

Full Code Listing # {#code}
===========================

Here is the complete code listing for this and the previous chapter.
Note that it is fully self-contained: you don't need LLVM or any
external libraries at all for this.

{% highlight python %} #!/usr/bin/env python

import re

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

class NumberExpressionNode(ExpressionNode): def **init**\ (self, value):
self.value = value

Expression class for referencing a variable, like "a".
======================================================

class VariableExpressionNode(ExpressionNode): def **init**\ (self,
name): self.name = name

Expression class for a binary operator.
=======================================

class BinaryOperatorExpressionNode(ExpressionNode): def **init**\ (self,
operator, left, right): self.operator = operator self.left = left
self.right = right

Expression class for function calls.
====================================

class CallExpressionNode(ExpressionNode): def **init**\ (self, callee,
args): self.callee = callee self.args = args

This class represents the "prototype" for a function, which captures its name,
==============================================================================

and its argument names (thus implicitly the number of arguments the function
============================================================================

takes).
=======

class PrototypeNode(object): def **init**\ (self, name, args): self.name
= name self.args = args

This class represents a function definition itself.
===================================================

class FunctionNode(object): def **init**\ (self, prototype, body):
self.prototype = prototype self.body = body

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
self.Handle(self.ParseDefinition, 'Parsed a function definition.')

def HandleExtern(self): self.Handle(self.ParseExtern, 'Parsed an
extern.')

def HandleTopLevelExpression(self): self.Handle(self.ParseTopLevelExpr,
'Parsed a top-level expression.')

def Handle(self, function, message): try: function() print message
except Exception, e: print 'Error:', e try: self.Next() # Skip for error
recovery. except: pass

Main driver code.
-----------------

def main(): # Install standard binary operators. # 1 is lowest possible
precedence. 40 is the highest. operator\_precedence = { '<': 10, '+':
20, '-': 20, '\*': 40 }

# Run the main "interpreter loop". while True: print 'ready>', try: raw
= raw\_input() except KeyboardInterrupt: return

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

if **name** == '**main**\ ': main() {% endhighlight %}

--------------

**`Next: Implementing Code Generation to LLVM
IR <PythonLangImpl3.html>`_**
