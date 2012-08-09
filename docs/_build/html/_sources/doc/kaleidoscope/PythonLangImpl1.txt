*************************************************
Chapter 1: Tutorial Introduction and the Lexer
*************************************************

Written by `Chris Lattner <mailto:sabre@nondot.org>`_ and `Max
Shawabkeh <http://max99x.com>`_

Introduction
=========

Welcome to the "Implementing a language with LLVM" tutorial. This
tutorial runs through the implementation of a simple language, showing
how fun and easy it can be. This tutorial will get you up and started as
well as help to build a framework you can extend to other languages. The
code in this tutorial can also be used as a playground to hack on other
LLVM specific things.

It is useful to point out ahead of time that this tutorial is really
about teaching compiler techniques and LLVM specifically, *not* about
teaching modern and sane software engineering principles. In practice,
this means that we'll take a number of shortcuts to simplify the
exposition. If you dig in and use the code as a basis for future
projects, fixing its deficiencies shouldn't be hard.

We've tried to put this tutorial together in a way that makes chapters
easy to skip over if you are already familiar with or are uninterested
in the various pieces. The structure of the tutorial is:

-  **`Chapter 1 <#language>`_: Introduction to the Kaleidoscope
   language, and the definition of its Lexer** -- This shows where we
   are going and the basic functionality that we want it to do. In order
   to make this tutorial maximally understandable and hackable, we
   choose to implement everything in Python instead of using lexer and
   parser generators. LLVM obviously works just fine with such tools,
   feel free to use one if you prefer.

-  **`Chapter 2 <PythonLangImpl2.html>`_: Implementing a Parser and
   AST** -- With the lexer in place, we can talk about parsing
   techniques and basic AST construction. This tutorial describes
   recursive descent parsing and operator precedence parsing. Nothing in
   Chapters 1 or 2 is LLVM-specific, the code doesn't even import the
   LLVM modules at this point. :)

-  **`Chapter 3 <PythonLangImpl3.html>`_: Code generation to LLVM IR**
   -- With the AST ready, we can show off how easy generation of LLVM IR
   really is.

-  **`Chapter 4 <PythonLangImpl4.html>`_: Adding JIT and Optimizer
   support** -- Because a lot of people are interested in using LLVM as
   a JIT, we'll dive right into it and show you the 3 lines it takes to
   add JIT support. LLVM is also useful in many other ways, but this is
   one simple and "sexy" way to shows off its power. :)

-  **`Chapter 5 <PythonLangImpl5.html>`_: Extending the Language:
   Control Flow** -- With the language up and running, we show how to
   extend it with control flow operations (if/then/else and a 'for'
   loop). This gives us a chance to talk about simple SSA construction
   and control flow.

-  **`Chapter 6 <PythonLangImpl6.html>`_: Extending the Language:
   User-defined Operators** -- This is a silly but fun chapter that
   talks about extending the language to let the user program define
   their own arbitrary unary and binary operators (with assignable
   precedence!). This lets us build a significant piece of the
   "language" as library routines.

-  **`Chapter 7 <PythonLangImpl7.html>`_: Extending the Language:
   Mutable Variables** -- This chapter talks about adding user-defined
   local variables along with an assignment operator. The interesting
   part about this is how easy and trivial it is to construct SSA form
   in LLVM: no, LLVM does *not* require your front-end to construct SSA
   form!

-  **`Chapter 8 <PythonLangImpl8.html>`_: Conclusion and other useful
   LLVM tidbits** -- This chapter wraps up the series by talking about
   potential ways to extend the language, but also includes a bunch of
   pointers to info about "special topics" like adding garbage
   collection support, exceptions, debugging, support for "spaghetti
   stacks", and a bunch of other tips and tricks.

By the end of the tutorial, we'll have written a bit less than 540 lines
of non-comment, non-blank, lines of code. With this small amount of
code, we'll have built up a very reasonable compiler for a non-trivial
language including a hand-written lexer, parser, AST, as well as code
generation support with a JIT compiler. While other systems may have
interesting "hello world" tutorials, I think the breadth of this
tutorial is a great testament to the strengths of LLVM and why you
should consider it if you're interested in language or compiler design.

A note about this tutorial: we expect you to extend the language and
play with it on your own. Take the code and go crazy hacking away at it,
compilers don't need to be scary creatures - it can be a lot of fun to
play with languages!

--------------

The Basic Language # {#language}
================================

This tutorial will be illustrated with a toy language that we'll call
"`Kaleidoscope <http://en.wikipedia.org/wiki/Kaleidoscope>`_\ " (derived
from "meaning beautiful, form, and view"). Kaleidoscope is a procedural
language that allows you to define functions, use conditionals, math,
etc. Over the course of the tutorial, we'll extend Kaleidoscope to
support the if/then/else construct, a for loop, user defined operators,
JIT compilation with a simple command line interface, etc.

Because we want to keep things simple, the only datatype in Kaleidoscope
is a 64-bit floating point type. As such, all values are implicitly
double precision and the language doesn't require type declarations.
This gives the language a very nice and simple syntax. For example, the
following simple example computes `Fibonacci
numbers <http://en.wikipedia.org/wiki/Fibonacci_number>`_:

{% highlight python %} # Compute the x'th fibonacci number. def fib(x)
if x < 3 then 1 else fib(x-1)+fib(x-2)

This expression will compute the 40th number.
=============================================

fib(40) {% endhighlight %}

We also allow Kaleidoscope to call into standard library functions (the
LLVM JIT makes this completely trivial). This means that you can use the
'extern' keyword to define a function before you use it (this is also
useful for mutually recursive functions). For example:

{% highlight python %} extern sin(arg); extern cos(arg); extern
atan2(arg1 arg2);

atan2(sin(0.4), cos(42)) {% endhighlight %}

A more interesting example is included in Chapter 6 where we write a
little Kaleidoscope application that
`displays <PythonLangImpl6.html#example>`_ a Mandelbrot Set at various
levels of magnification.

Lets dive into the implementation of this language!

--------------

The Lexer # {#lexer}
====================

When it comes to implementing a language, the first thing needed is the
ability to process a text file and recognize what it says. The
traditional way to do this is to use a
`lexer <http://en.wikipedia.org/wiki/Lexical_analysis>`_" (aka
'scanner') to break the input up into "tokens". Each token returned by
the lexer includes a token type and potentially some metadata (e.g. the
numeric value of a number). First, we define the possibilities:

{% highlight python %} # The lexer yields one of these types for each
token. class EOFToken(object): pass

class DefToken(object): pass

class ExternToken(object): pass

class IdentifierToken(object): def **init**\ (self, name): self.name =
name

class NumberToken(object): def **init**\ (self, value): self.value =
value

class CharacterToken(object): def **init**\ (self, char): self.char =
char def **eq**\ (self, other): return isinstance(other, CharacterToken)
and self.char == other.char def **ne**\ (self, other): return not self
== other {% endhighlight %}

Each token yielded by our lexer will be of one of the above types. For
simple tokens that are always the same, like the "def" keyword, the
lexer will yield ``DefToken()``>. Identifiers, numbers and characters,
on the other hand, have extra data, so when the lexer encounteres the
number 123.45, it will emit it as ``NumberToken(123.45)``. An identifier
``foo`` will be emitted as ``IdentifierToken('foo')``. And finally, an
unknown character like '+' will be returned as ``CharacterToken('+')``.
You may notice that we overload the equality and inequality operators
for the characters; this will later simplify character comparisons in
the parser code.

The actual implementation of the lexer is a single function called
``Tokenize``, which takes a string and
`yields <http://docs.python.org/reference/simple_stmts.html#the-yield-statement>`_
tokens. For simplicity, we will use `regular
expressions <http://docs.python.org/library/re.html>`_ to parse out the
tokens. This is terribly inefficient, but perfectly sufficient for our
needs.

First, we define the regular expressions for our tokens. Numbers and
strings of digits, optionally followed by a period and another string of
digits. Identifiers (and keywords) are alphanumeric string starting with
a letter and comments are anything between a hash (``#``) and the end of
the line.

{% highlight python %} import re

...

Regular expressions that tokens and comments of our language.
=============================================================

REGEX\_NUMBER = re.compile('[0-9]+(?:.[0-9]+)?') REGEX\_IDENTIFIER =
re.compile('[a-zA-Z][a-zA-Z0-9]\ *') REGEX\_COMMENT = re.compile('#.*')

{% endhighlight %}

Next, let's start defining the ``Tokenize`` function itself. The first
thing we need to do is set up a loop that scans the string, while
ignoring whitespace between tokens:

{% highlight python %} def Tokenize(string): while string: # Skip
whitespace. if string[0].isspace(): string = string[1:] continue

::

    ...

{% endhighlight %}

Next we want to find out what the next token is. For this we run the
regexes we defined above on the remainder of the string. To simplify the
rest of the code, we run all three regexes each time. As mentioned
above, inefficiencies are ignored for the purpose of this tutorial:

{% highlight python %} # Run regexes. comment\_match =
REGEX\_COMMENT.match(string) number\_match = REGEX\_NUMBER.match(string)
identifier\_match = REGEX\_IDENTIFIER.match(string) {% endhighlight %}

Now se check if any of the regexes matched. For comments, we simply
ignore the captured match:

{% highlight python %} # Check if any of the regexes matched and yield
the appropriate result. if comment\_match: comment =
comment\_match.group(0) string = string[len(comment):] {% endhighlight
python %}

For numbers, we yield the captured match, converted to a float and
tagged with the appropriate token type:

{% highlight python %} elif number\_match: number =
number\_match.group(0) yield NumberToken(float(number)) string =
string[len(number):] {% endhighlight %}

The identifier case is a little more complex. We have to check for
keywords to decide whether we have captured an identifier or a keyword:

{% highlight python %} elif identifier\_match: identifier =
identifier\_match.group(0) # Check if we matched a keyword. if
identifier == 'def': yield DefToken() elif identifier == 'extern': yield
ExternToken() else: yield IdentifierToken(identifier) string =
string[len(identifier):] {% endhighlight %}

Finally, if we haven't recognized a comment, a number of an identifier,
we yield the current character as an "unknown character" token. This is
used, for example, for operators like ``+`` or ``*``:

{% highlight python %} else: # Yield the unknown character. yield
CharacterToken(string[0]) string = string[1:] {% endhighlight %}

Once we're done with the loop, we return a final end-of-file token:

{% highlight python %} yield EOFToken() {% endhighlight %}

With this, we have the complete lexer for the basic Kaleidoscope
language (the `full code listing <PythonLangImpl2.html#code>`_ for the
Lexer is available in the `next chapter <PythonLangImpl2.html>`_ of the
tutorial). Next we'll `build a simple parser that uses this to build an
Abstract Syntax Tree <PythonLangImpl2.html>`_. When we have that, we'll
include a driver so that you can use the lexer and parser together.

--------------

**`Next: Implementing a Parser and AST <PythonLangImpl2.html>`_**
