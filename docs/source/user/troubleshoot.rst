
.. _numba-troubleshooting:

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
is listed there and still fails compiling, please
:ref:`report a bug <report-numba-bugs>`.

When Numba tries to compile your code it first tries to work out the types of
all the variables in use, this is so it can generate a type specific
implementation of your code that can be compiled down to machine code. A common
reason for Numba failing to compile (especially in :term:`nopython mode`) is a
type inference failure, essentially Numba cannot work out what the type of all
the variables in your code should be. 

For example, let's consider this trivial function::

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
    File "<path>/numba/numba/dispatcher.py", line 339, in _compile_for_args
        reraise(type(e), e, None)
    File "<path>/numba/numba/six.py", line 658, in reraise
        raise value.with_traceback(tb)
    numba.errors.TypingError: Failed at nopython (nopython frontend)
    Invalid use of + with parameters (int64, tuple(int64 x 1))
    Known signatures:
    * (int64, int64) -> int64
    * (int64, uint64) -> int64
    * (uint64, int64) -> int64
    * (uint64, uint64) -> uint64
    * (float32, float32) -> float32
    * (float64, float64) -> float64
    * (complex64, complex64) -> complex64
    * (complex128, complex128) -> complex128
    * (uint16,) -> uint64
    * (uint8,) -> uint64
    * (uint64,) -> uint64
    * (uint32,) -> uint64
    * (int16,) -> int64
    * (int64,) -> int64
    * (int8,) -> int64
    * (int32,) -> int64
    * (float32,) -> float32
    * (float64,) -> float64
    * (complex64,) -> complex64
    * (complex128,) -> complex128
    * parameterized
    [1] During: typing of intrinsic-call at <stdin> (3)

    File "<stdin>", line 3:

The error message helps you find out what went wrong:
"Invalid use of + with parameters (int64, tuple(int64 x 1))" is to be
interpreted as "Numba encountered an addition of variables typed as integer
and 1-tuple of integer, respectively, and doesn't know about any such
operation".

Note that if you allow object mode::

    @jit
    def g(x, y):
        return x + y

compilation will succeed and the compiled function will raise at runtime as
Python would do::

   >>> g(1, (2,))
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   TypeError: unsupported operand type(s) for +: 'int' and 'tuple'


My code has a type unification problem
======================================

Another common reason for Numba not being able to compile your code is that it
cannot statically determine the return type of a function. The most likely
cause of this is the return type depending on a value that is available only at
runtime. Again, this is most often problematic when using
:term:`nopython mode`. The concept of type unification is simply trying to find
a type in which two variables could safely be represented. For example a 64 bit
float and a 64 bit complex number could both be represented in a 128 bit complex
number.

As an example of type unification failure, this function has a return type that
is determined at runtime based on the value of `x`::

    In [1]: from numba import jit

    In [2]: @jit(nopython=True)
    ...: def f(x):
    ...:     if x > 10:
    ...:         return (1,)
    ...:     else:
    ...:         return 1
    ...:     

    In [3]: f(10)

Trying to execute this function, errors out as follows:: 

    TypingError: Failed at nopython (nopython frontend)
    Can't unify return type from the following types: tuple(int64 x 1), int64
    Return of: IR name '$8.2', type '(int64 x 1)', location: 
    File "<ipython-input-2-51ef1cc64bea>", line 4:
    def f(x):
        <source elided>
        if x > 10:
            return (1,)
            ^
    Return of: IR name '$12.2', type 'int64', location: 
    File "<ipython-input-2-51ef1cc64bea>", line 6:
    def f(x):
        <source elided>
        else:
            return 1

The error message "Can't unify return type from the following types:
tuple(int64 x 1), int64" should be read as "Numba cannot find a type that
can safely represent a 1-tuple of integer and an integer".


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


Debugging JIT compiled code with GDB
====================================

Setting the ``debug`` keyword argument in the ``jit`` decorator
(e.g. ``@jit(debug=True)``) enables the emission of debug info in the jitted
code.  To debug, GDB version 7.0 or above is required.  Currently, the following
debug info is available:

* Function name will be shown in the backtrace.  But, no type information.
* Source location (filename and line number) is available.  For example,
  user can set break point by the absolute filename and line number;
  e.g. ``break /path/to/myfile.py:6``.
* Local variables in the current function can be shown with ``info locals``.
* Type of variable with ``whatis myvar``.
* Value of variable with ``print myvar`` or ``display myvar``.

  * Simple numeric types, i.e. int, float and double, are shown in their
    native representation.  But, integers are assumed to be signed.
  * Other types are shown as sequence of bytes.

Known issues:

* Stepping depends heavily on optimization level.

  * At full optimization (equivalent to O3), most of the variables are
    optimized out.
  * With no optimization (e.g. ``NUMBA_OPT=0``), source location jumps around
    when stepping through the code.
  * At O1 optimization (e.g. ``NUMBA_OPT=1``), stepping is stable but some
    variables are optimized out.

* Memory consumption increases significantly with debug info enabled.
  The compiler emits extra information (`DWARF <http://www.dwarfstd.org/>`_)
  along with the instructions.  The emitted object code can be 2x bigger with
  debug info.

Internal details:

* Since Python semantics allow variables to bind to value of different types,
  Numba internally creates multiple versions of the variable for each type.
  So for code like::

    x = 1         # type int
    x = 2.3       # type float
    x = (1, 2, 3) # type 3-tuple of int

  Each assignments will store to a different variable name.  In the debugger,
  the variables will be ``x``, ``x$1`` and ``x$2``.  (In the Numba IR, they are
  ``x``, ``x.1`` and ``x.2``.)

* When debug is enabled, inlining of the function is disabled.

Example debug usage
-------------------

The python source:

.. code-block:: python
  :linenos:

  from numba import njit

  @njit(debug=True)
  def foo(a):
      b = a + 1
      c = a * 2.34
      d = (a, b, c)
      print(a, b, c, d)

  r= foo(123)
  print(r)

In the terminal:

.. code-block:: none
  :emphasize-lines: 1, 8, 13, 15, 20, 25, 27, 29

  $ NUMBA_OPT=1 gdb -q python
  Reading symbols from python...done.
  (gdb) break /home/user/chk_debug.py:5
  No source file named /home/user/chk_debug.py.
  Make breakpoint pending on future shared library load? (y or [n]) y

  Breakpoint 1 (/home/user/chk_debug.py:5) pending.
  (gdb) run chk_debug.py
  Starting program: /home/user/miniconda/bin/python chk_debug.py
  ...
  Breakpoint 1, __main__::foo$241(long long) () at chk_debug.py:5
  5	    b = a + 1
  (gdb) n
  6	    c = a * 2.34
  (gdb) bt
  #0  __main__::foo$241(long long) () at chk_debug.py:6
  #1  0x00007ffff7fec47c in cpython::__main__::foo$241(long long) ()
  #2  0x00007fffeb7976e2 in call_cfunc (locals=0x0, kws=0x0, args=0x7fffeb486198,
  ...
  (gdb) info locals
  a = 0
  d = <error reading variable d (DWARF-2 expression error: `DW_OP_stack_value' operations must be used either alone or in conjunction with DW_OP_piece or DW_OP_bit_piece.)>
  c = 0
  b = 124
  (gdb) whatis b
  type = i64
  (gdb) whatis d
  type = {i64, i64, double}
  (gdb) print b
  $2 = 124

Globally override debug setting
-------------------------------

It is possible to enable debug for the full application by setting environment
variable ``NUMBA_DEBUGINFO=1``.  This sets the default value of the ``debug``
option in ``jit``.  Debug can be turned off on individual functions by setting
``debug=False``.

Beware that enabling debug info significantly increases the memory consumption
for each compiled function.  For large application, this may cause out-of-memory
error.

Debugging CUDA Python code
==========================

Using the simulator
-------------------

CUDA Python code can be run in the Python interpreter using the CUDA Simulator,
allowing it to be debugged with the Python debugger or with print statements. To
enable the CUDA simulator, set the environment variable
:envvar:`NUMBA_ENABLE_CUDASIM` to 1. For more information on the CUDA Simulator,
see :ref:`the CUDA Simulator documentation <simulator>`.


Debug Info
----------

By setting the ``debug`` argument to ``cuda.jit`` to ``True``
(``@cuda.jit(debug=True)``), Numba will emit source location in the compiled
CUDA code.  Unlike the CPU target, only filename and line information are
available, but no variable type information is emitted.  The information
is sufficient to debug memory error with
`cuda-memcheck <http://docs.nvidia.com/cuda/cuda-memcheck/index.html>`_.

For example, given the following cuda python code:

.. code-block:: python
  :linenos:

  import numpy as np
  from numba import cuda

  @cuda.jit(debug=True)
  def foo(arr):
      arr[cuda.threadIdx.x] = 1

  arr = np.arange(30)
  foo[1, 32](arr)   # more threads than array elements

We can use ``cuda-memcheck`` to find the memory error:

.. code-block:: none

  $ cuda-memcheck python chk_cuda_debug.py
  ========= CUDA-MEMCHECK
  ========= Invalid __global__ write of size 8
  =========     at 0x00000148 in /home/user/chk_cuda_debug.py:6:cudapy::__main__::foo$241(Array<__int64, int=1, C, mutable, aligned>)
  =========     by thread (31,0,0) in block (0,0,0)
  =========     Address 0x500a600f8 is out of bounds
  ...
  =========
  ========= Invalid __global__ write of size 8
  =========     at 0x00000148 in /home/user/chk_cuda_debug.py:6:cudapy::__main__::foo$241(Array<__int64, int=1, C, mutable, aligned>)
  =========     by thread (30,0,0) in block (0,0,0)
  =========     Address 0x500a600f0 is out of bounds
  ...
