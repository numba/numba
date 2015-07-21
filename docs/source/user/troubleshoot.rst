
.. _troubleshooting:

========================
Troubleshooting and tips
========================

.. _what-to-compile:

What to compile
===============

The general recommendation is that you should only try to compile the
critical paths in your code.  If you have a piece of performance-critical
computational code amongst some higher-level code, you may factor out
the performance-critical code in a separate function and compile the
separate function with Numba.  Letting Numba focus on that small piece
of performance-critical code has several advantages:

* it reduces the risk of hitting unsupported features;
* it reduces the compilation times;
* it allows you to evolve the higher-level code which is outside of the
  compiled function much easier.

.. _code-doesnt-compile:

My code doesn't compile
=======================

There can be various reasons why Numba cannot compile your code, and raises
an error instead.  One common reason is that your code relies on an
unsupported Python feature, especially in :term:`nopython mode`.
Please see the list of :ref:`pysupported`.  If you find something that
is listed there and still fails compiling, please :ref:`report a bug <report-bugs>`.

The other reason is that you asked for :term:`nopython mode`, and type
inference has failed on some piece of your code.  For example, let's
consider this trivial function::

   @jit(nopython=True)
   def f(x, y):
       return x + y

If you call it with two numbers, Numba is able to infer the types properly::

   >>> f(1, 2)
   3

If however you call it with a tuple and a number, Numba is unable to say
what the result of adding a tuple and number is, and therefore compilation
errors out::

   >>> f(1, (2,))
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
       [...]
     File "/home/antoine/numba/numba/typeinfer.py", line 242, in resolve
       raise TypingError(msg, loc=self.loc)
   numba.typeinfer.TypingError: Failed at nopython frontend
   Undeclared +(int64, (int32 x 1))
   File "<stdin>", line 2

The error message helps you find out what went wrong:
"Undeclared +(int64, (int32 x 1))" is to be interpreted as "Numba encountered
an addition of variables typed as integer and 1-tuple of integer, respectively,
and doesn't know about any such operation".

Note that if you allow object mode, compilation will succeed and the
compiled function will raise at runtime as Python would do::

   >>> g(1, (2,))
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   TypeError: unsupported operand type(s) for +: 'int' and 'tuple'

The compiled code is too slow
=============================

The most common reason for slowness of a compiled JIT function is that
compiling in :term:`nopython mode` has failed and the Numba compiler has
fallen back to :term:`object mode`.  :term:`object mode` currently provides
little to no speedup compared to regular Python interpretation, and its
main point is to allow an internal optimization known as
:term:`loop-lifting`: this optimization will allow to compile inner
loops in :term:`nopython mode` regardless of what code surrounds those
inner loops.

To find out if type inference succeeded on your function, you can use
the :meth:`~Dispatcher.inspect_types` method on the compiled function.

For example, let's take the following function::

   @jit
   def f(a, b):
       s = a + float(b)
       return s

When called with numbers, this function should be fast as Numba is able
to convert number types to floating-point numbers.  Let's see::

   >>> f(1, 2)
   3.0
   >>> f.inspect_types()
   f (int64, int64)
   --------------------------------------------------------------------------------
   # --- LINE 7 ---

   @jit

   # --- LINE 8 ---

   def f(a, b):

       # --- LINE 9 ---
       # label 0
       #   a.1 = a  :: int64
       #   del a
       #   b.1 = b  :: int64
       #   del b
       #   $0.2 = global(float: <class 'float'>)  :: Function(<class 'float'>)
       #   $0.4 = call $0.2(b.1, )  :: (int64,) -> float64
       #   del b.1
       #   del $0.2
       #   $0.5 = a.1 + $0.4  :: float64
       #   del a.1
       #   del $0.4
       #   s = $0.5  :: float64
       #   del $0.5

       s = a + float(b)

       # --- LINE 10 ---
       #   $0.7 = cast(value=s)  :: float64
       #   del s
       #   return $0.7

       return s

Without trying to understand too much of the Numba intermediate representation,
it is still visible that all variables and temporary values have had their
types inferred properly: for example *a* has the type ``int64``, *$0.5* has
the type ``float64``, etc.

However, if *b* is passed as a string, compilation will fall back on object
mode as the float() constructor with a string is currently not supported
by Numba::

   >>> f(1, "2")
   3.0
   >>> f.inspect_types()
   [... snip annotations for other signatures, see above ...]
   ================================================================================
   f (int64, str)
   --------------------------------------------------------------------------------
   # --- LINE 7 ---

   @jit

   # --- LINE 8 ---

   def f(a, b):

       # --- LINE 9 ---
       # label 0
       #   a.1 = a  :: pyobject
       #   del a
       #   b.1 = b  :: pyobject
       #   del b
       #   $0.2 = global(float: <class 'float'>)  :: pyobject
       #   $0.4 = call $0.2(b.1, )  :: pyobject
       #   del b.1
       #   del $0.2
       #   $0.5 = a.1 + $0.4  :: pyobject
       #   del a.1
       #   del $0.4
       #   s = $0.5  :: pyobject
       #   del $0.5

       s = a + float(b)

       # --- LINE 10 ---
       #   $0.7 = cast(value=s)  :: pyobject
       #   del s
       #   return $0.7

       return s

Here we see that all variables end up typed as ``pyobject``.  This means
that the function was compiled in object mode and values are passed
around as generic Python objects, without Numba trying to look into them
to reason about their raw values.  This is a situation you want to avoid
when caring about the speed of your code.

There are several ways of understanding why a function fails to
compile in nopython mode:

* pass *nopython=True*, which will raise an error indicating what went wrong
  (see above :ref:`code-doesnt-compile`);
* enable warnings by setting the :envvar:`NUMBA_WARNINGS` environment
  variable; for example with the ``f()`` function above::

      >>> f(1, 2)
      3.0
      >>> f(1, "2")
      example.py:7: NumbaWarning: Function "f" failed type inference: Internal error at <numba.typeinfer.CallConstrain object at 0x7f6b8dd24550>:
      float() only support for numbers
      File "example.py", line 9
        @jit
      example.py:7: NumbaWarning: Function "f" was compiled in object mode without forceobj=True.
        @jit
      3.0

Disabling JIT compilation
=========================

In order to debug code, it is possible to disable JIT compilation, which makes
the ``jit`` decorator (and the decorators ``njit`` and ``autojit``) act as if
they perform no operation, and the invocation of decorated functions calls the
original Python function instead of a compiled version. This can be toggled by
setting the :envvar:`NUMBA_DISABLE_JIT` enviroment variable to ``1``.

When this mode is enabled, the ``vectorize`` and ``guvectorize`` decorators will
still result in compilation of a ufunc, as there is no straightforward pure
Python implementation of these functions.

Debugging CUDA Python code
==========================

CUDA Python code can be run in the Python interpreter using the CUDA Simulator,
allowing it to be debugged with the Python debugger or with print statements. To
enable the CUDA simulator, set the environment variable
:envvar:`NUMBA_ENABLE_CUDASIM` to 1. For more information on the CUDA Simulator,
see :ref:`the CUDA Simulator documentation <simulator>`.
