
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

.. _code-has-untyped-list:

My code has an untyped list problem
===================================

As :ref:`noted previously <code-doesnt-compile>` the first part of Numba
compiling your code involves working out what the types of all the variables
are. In the case of lists, a list must contain items that are of the same type
or can be empty if the type can be inferred from some later operation. What is
not possible is to have a list which is defined as empty and has no inferable
type (i.e. an untyped list).

For example, this is using a list of a known type::

    from numba import jit
    @jit(nopython=True)
    def f():
        return [1, 2, 3] # this list is defined on construction with `int` type

This is using an empty list, but the type can be inferred::

    from numba import jit
    @jit(nopython=True)
    def f(x):
        tmp = [] # defined empty
        for i in range(x):
            tmp.append(i) # list type can be inferred from the type of `i`
        return tmp

This is using an empty list and the type cannot be inferred::

    from numba import jit
    @jit(nopython=True)
    def f(x):
        tmp = [] # defined empty
        return (tmp, x) # ERROR: the type of `tmp` is unknown

Whilst slightly contrived, if you need an empty list and the type cannot be
inferred but you know what type you want the list to be, this "trick" can be
used to instruct the typing mechanism::

    from numba import jit
    import numpy as np
    @jit(nopython=True)
    def f(x):
        # define empty list, but instruct that the type is np.complex64
        tmp = [np.complex64(x) for x in range(0)]
        return (tmp, x) # the type of `tmp` is known, but it is still empty

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

If a function fails to compile in ``nopython`` mode warnings will be emitted
with explanation as to why compilation failed. For example with the ``f()``
function above (slightly edited for documentation purposes)::

    >>> f(1, 2)
    3.0
    >>> f(1, "2")
    example.py:7: NumbaWarning:
    Compilation is falling back to object mode WITH looplifting enabled because Function "f" failed type inference due to: Invalid use of Function(<class 'float'>) with argument(s) of type(s): (unicode_type)
    * parameterized
    In definition 0:
        TypeError: float() only support for numbers
        raised from <path>/numba/typing/builtins.py:880
    In definition 1:
        TypeError: float() only support for numbers
        raised from <path>/numba/typing/builtins.py:880
    This error is usually caused by passing an argument of a type that is unsupported by the named function.
    [1] During: resolving callee type: Function(<class 'float'>)
    [2] During: typing of call at example.py (9)


    File "example.py", line 9:
    def f(a, b):
        s = a + float(b)
        ^

    <path>/numba/compiler.py:722: NumbaWarning: Function "f" was compiled in object mode without forceobj=True.

    File "example.py", line 8:
    @jit
    def f(a, b):
    ^

    3.0


Disabling JIT compilation
=========================

In order to debug code, it is possible to disable JIT compilation, which makes
the ``jit`` decorator (and the ``njit`` decorator) act as if
they perform no operation, and the invocation of decorated functions calls the
original Python function instead of a compiled version. This can be toggled by
setting the :envvar:`NUMBA_DISABLE_JIT` enviroment variable to ``1``.

When this mode is enabled, the ``vectorize`` and ``guvectorize`` decorators will
still result in compilation of a ufunc, as there is no straightforward pure
Python implementation of these functions.


.. _debugging-jit-compiled-code:

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

Using Numba's direct ``gdb`` bindings in ``nopython``  mode
===========================================================

Numba (version 0.42.0 and later) has some additional functions relating to
``gdb`` support for CPUs that make it easier to debug programs. All the ``gdb``
related functions described in the following work in the same manner
irrespective of whether they are called from the standard CPython interpreter or
code compiled in either :term:`nopython mode` or :term:`object mode`.

.. note:: This feature is experimental!

.. warning:: This feature does unexpected things if used from Jupyter or
             alongside the ``pdb`` module. It's behaviour is harmless, just hard
             to predict!

Set up
------
Numba's ``gdb`` related functions make use of a ``gdb`` binary, the location and
name of this binary can be configured via the :envvar:`NUMBA_GDB_BINARY`
environment variable if desired.

.. note:: Numba's ``gdb`` support requires the ability for ``gdb`` to attach to
          another process. On some systems (notably Ubuntu Linux) default
          security restrictions placed on ``ptrace`` prevent this from being
          possible. This restriction is enforced at the system level by the
          Linux security module `Yama`. Documentation for this module and the
          security implications of making changes to its behaviour can be found
          in the `Linux Kernel documentation <https://www.kernel.org/doc/Documentation/admin-guide/LSM/Yama.rst>`_.
          The `Ubuntu Linux security documentation <https://wiki.ubuntu.com/Security/Features#ptrace>`_
          discusses how to adjust the behaviour of `Yama` on with regards to
          ``ptrace_scope`` so as to permit the required behaviour.

Basic ``gdb`` support
---------------------

.. warning:: Calling :func:`numba.gdb` and/or :func:`numba.gdb_init` more than
             once in the same program is not advisable, unexpected things may
             happen. If multiple breakpoints are desired within a program,
             launch ``gdb`` once via :func:`numba.gdb` or :func:`numba.gdb_init`
             and then use :func:`numba.gdb_breakpoint` to register additional
             breakpoint locations.

The most simple function for adding ``gdb`` support is :func:`numba.gdb`, which,
at the call location, will:

* launch ``gdb`` and attach it to the running process.
* create a breakpoint at the site of the :func:`numba.gdb()` function call, the
  attached ``gdb`` will pause execution here awaiting user input.

use of this functionality is best motivated by example, continuing with the
example used above:

.. code-block:: python
  :linenos:

  from numba import njit, gdb

  @njit(debug=True)
  def foo(a):
      b = a + 1
      gdb() # instruct Numba to attach gdb at this location and pause execution
      c = a * 2.34
      d = (a, b, c)
      print(a, b, c, d)

  r= foo(123)
  print(r)

In the terminal (``...`` on a line by itself indicates output that is not
presented for brevity):

.. code-block:: none
    :emphasize-lines: 1, 15, 20, 31, 33, 35, 37, 39

    $ NUMBA_OPT=0 python demo_gdb.py
    Attaching to PID: 27157
    GNU gdb (GDB) Red Hat Enterprise Linux 8.0.1-36.el7
    ...
    Attaching to process 27157
    ...
    Reading symbols from <elided for brevity> ...done.
    0x00007f0380c31550 in __nanosleep_nocancel () at ../sysdeps/unix/syscall-template.S:81
    81      T_PSEUDO (SYSCALL_SYMBOL, SYSCALL_NAME, SYSCALL_NARGS)
    Breakpoint 1 at 0x7f036ac388f0: file numba/_helperlib.c, line 1090.
    Continuing.

    Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    (gdb) s
    Single stepping until exit from function _ZN5numba7targets8gdb_hook8hook_gdb12$3clocals$3e8impl$242E5Tuple,
    which has no line number information.
    __main__::foo$241(long long) () at demo_gdb.py:7
    7           c = a * 2.34
    (gdb) l
    2
    3       @njit(debug=True)
    4       def foo(a):
    5           b = a + 1
    6           gdb() # instruct Numba to attach gdb at this location and pause execution
    7           c = a * 2.34
    8           d = (a, b, c)
    9           print(a, b, c, d)
    10
    11      r= foo(123)
    (gdb) p a
    $1 = 123
    (gdb) p b
    $2 = 124
    (gdb) p c
    $3 = 0
    (gdb) n
    8           d = (a, b, c)
    (gdb) p c
    $4 = 287.81999999999999

It can be seen in the above example that execution of the code is paused at the
location of the ``gdb()`` function call at end of the ``numba_gdb_breakpoint``
function (this is the Numba internal symbol registered as breakpoint with
``gdb``). Issuing a ``step`` at this point moves to the stack frame of the
compiled Python source. From there, it can be seen that the variables ``a`` and
``b`` have been evaluated but ``c`` has not, as demonstrated by printing their
values, this is precisely as expected given the location of the ``gdb()`` call.
Issuing a ``next`` then evaluates line ``7`` and ``c`` is assigned a value as
demonstrated by the final print.

Running with ``gdb`` enabled
----------------------------

The functionality provided by :func:`numba.gdb` (launch and attach ``gdb`` to
the executing process and pause on a breakpoint) is also available as two
separate functions:

* :func:`numba.gdb_init` this function injects code at the call site to launch
  and attach ``gdb`` to the executing process but does not pause execution.
* :func:`numba.gdb_breakpoint` this function injects code at the call site that
  will call the special ``numba_gdb_breakpoint`` function that is registered as
  a breakpoint in Numba's ``gdb`` support. This is demonstrated in the next
  section.

This functionality enables more complex debugging capabilities. Again, motivated
by example, debugging a 'segfault' (memory access violation signalling
``SIGSEGV``):

.. code-block:: python
  :linenos:

    from numba import njit, gdb_init
    import numpy as np

    @njit(debug=True)
    def foo(a, index):
        gdb_init() # instruct Numba to attach gdb at this location, but not to pause execution
        b = a + 1
        c = a * 2.34
        d = c[index] # access an address that is a) invalid b) out of the page
        print(a, b, c, d)

    bad_index = int(1e9) # this index is invalid
    z = np.arange(10)
    r = foo(z, bad_index)
    print(r)

In the terminal (``...`` on a line by itself indicates output that is not
presented for brevity):

.. code-block:: none
    :emphasize-lines: 1, 15, 17, 19, 21

    $ python demo_gdb_segfault.py
    Attaching to PID: 5444
    GNU gdb (GDB) Red Hat Enterprise Linux 8.0.1-36.el7
    ...
    Attaching to process 5444
    ...
    Reading symbols from <elided for brevity> ...done.
    0x00007f8d8010a550 in __nanosleep_nocancel () at ../sysdeps/unix/syscall-template.S:81
    81      T_PSEUDO (SYSCALL_SYMBOL, SYSCALL_NAME, SYSCALL_NARGS)
    Breakpoint 1 at 0x7f8d6a1118f0: file numba/_helperlib.c, line 1090.
    Continuing.

    0x00007fa7b810a41f in __main__::foo$241(Array<long long, 1, C, mutable, aligned>, long long) () at demo_gdb_segfault.py:9
    9           d = c[index] # access an address that is a) invalid b) out of the page
    (gdb) p index
    $1 = 1000000000
    (gdb) p c
    $2 = "p\202\017\364\371U\000\000\000\000\000\000\000\000\000\000\n\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\240\202\017\364\371U\000\000\n\000\000\000\000\000\000\000\b\000\000\000\000\000\000"
    (gdb) whatis c
    type = {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}
    (gdb) x /32xb c
    0x7ffd56195068: 0x70    0x82    0x0f    0xf4    0xf9    0x55    0x00    0x00
    0x7ffd56195070: 0x00    0x00    0x00    0x00    0x00    0x00    0x00    0x00
    0x7ffd56195078: 0x0a    0x00    0x00    0x00    0x00    0x00    0x00    0x00
    0x7ffd56195080: 0x08    0x00    0x00    0x00    0x00    0x00    0x00    0x00


In the ``gdb`` output it can be noted that the ``numba_gdb_breakpoint`` function
was registered as a breakpoint (its symbol is in ``numba/_helperlib.c``),
that a ``SIGSEGV`` signal was caught, and the line in which the access violation
occurred is printed.

Continuing the example as a debugging session demonstration, first ``index``
can be printed, and it is evidently 1e9. Printing ``c`` gives a lot of bytes, so
the type needs looking up. The type of ``c`` shows the layout for the array
``c`` based on its ``DataModel`` (look in the Numba source
``numba.datamodel.models`` for the layouts, the ``ArrayModel`` is presented
below for ease).

.. code-block:: python

    class ArrayModel(StructModel):
        def __init__(self, dmm, fe_type):
            ndim = fe_type.ndim
            members = [
                ('meminfo', types.MemInfoPointer(fe_type.dtype)),
                ('parent', types.pyobject),
                ('nitems', types.intp),
                ('itemsize', types.intp),
                ('data', types.CPointer(fe_type.dtype)),
                ('shape', types.UniTuple(types.intp, ndim)),
                ('strides', types.UniTuple(types.intp, ndim)),
            ]

The type inspected from ``gdb``
(``type = {i8*, i8*, i64, i64, double*, [1 x i64], [1 x i64]}``) corresponds
directly to the members of the ``ArrayModel``. Given the segfault came from an
invalid access it would be informative to check the number of items in the array
and compare that to the index requested.

Examining the memory of ``c``, (``x /32xb c``), the first 16 bytes are the two
``i8*`` corresponding to the ``meminfo`` pointer and the ``parent``
``pyobject``. The next two groups of 8 bytes are ``i64``/``intp`` types
corresponding to ``nitems`` and ``itemsize`` respectively. Evidently their
values are ``0x0a`` and ``0x08``, this makes sense as the input array ``a`` has
10 elements and is of type ``int64`` which is 8 bytes wide. It's therefore clear
that the segfault comes from an invalid access of index ``1000000000`` in an
array containing ``10`` items.

Adding breakpoints to code
--------------------------

The next example demonstrates using multiple breakpoints that are defined
through the invocation of the :func:`numba.gdb_breakpoint` function:

.. code-block:: python
  :linenos:

  from numba import njit, gdb_init, gdb_breakpoint

  @njit(debug=True)
  def foo(a):
      gdb_init() # instruct Numba to attach gdb at this location
      b = a + 1
      gdb_breakpoint() # instruct gdb to break at this location
      c = a * 2.34
      d = (a, b, c)
      gdb_breakpoint() # and to break again at this location
      print(a, b, c, d)

  r= foo(123)
  print(r)

In the terminal (``...`` on a line by itself indicates output that is not
presented for brevity):

.. code-block:: none
    :emphasize-lines: 1, 17, 20, 31, 33, 35, 40, 42

    $ NUMBA_OPT=0 python demo_gdb_breakpoints.py
    Attaching to PID: 20366
    GNU gdb (GDB) Red Hat Enterprise Linux 8.0.1-36.el7
    ...
    Attaching to process 20366
    Reading symbols from <elided for brevity> ...done.
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    Reading symbols from /lib64/libc.so.6...Reading symbols from /usr/lib/debug/usr/lib64/libc-2.17.so.debug...done.
    0x00007f631db5e550 in __nanosleep_nocancel () at ../sysdeps/unix/syscall-template.S:81
    81      T_PSEUDO (SYSCALL_SYMBOL, SYSCALL_NAME, SYSCALL_NARGS)
    Breakpoint 1 at 0x7f6307b658f0: file numba/_helperlib.c, line 1090.
    Continuing.

    Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    (gdb) step
    __main__::foo$241(long long) () at demo_gdb_breakpoints.py:8
    8           c = a * 2.34
    (gdb) l
    3       @njit(debug=True)
    4       def foo(a):
    5           gdb_init() # instruct Numba to attach gdb at this location
    6           b = a + 1
    7           gdb_breakpoint() # instruct gdb to break at this location
    8           c = a * 2.34
    9           d = (a, b, c)
    10          gdb_breakpoint() # and to break again at this location
    11          print(a, b, c, d)
    12
    (gdb) p b
    $1 = 124
    (gdb) p c
    $2 = 0
    (gdb) continue
    Continuing.

    Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    (gdb) step
    11          print(a, b, c, d)
    (gdb) p c
    $3 = 287.81999999999999

From the ``gdb`` output it can be seen that execution paused at line 8 as a
breakpoint was hit, and after a ``continue`` was issued, it broke again at line
11 where the next breakpoint was hit.

Debugging in parallel regions
-----------------------------

The follow example is quite involved, it executes with ``gdb`` instrumentation
from the outset as per the example above, but it also uses threads and makes use
of the breakpoint functionality. Further, the last iteration of the parallel
section calls the function ``work``, which is actually just a binding to
``glibc``'s ``free(3)`` in this case, but could equally be some involved
function that is presenting a segfault for unknown reasons.

.. code-block:: python
  :linenos:

    from numba import njit, prange, gdb_init, gdb_breakpoint
    import ctypes

    def get_free():
        lib = ctypes.cdll.LoadLibrary('libc.so.6')
        free_binding = lib.free
        free_binding.argtypes = [ctypes.c_void_p,]
        free_binding.restype = None
        return free_binding

    work = get_free()

    @njit(debug=True, parallel=True)
    def foo():
        gdb_init() # instruct Numba to attach gdb at this location, but not to pause execution
        counter = 0
        n = 9
        for i in prange(n):
            if i > 3 and i < 8: # iterations 4, 5, 6, 7 will break here
                gdb_breakpoint()

            if i == 8: # last iteration segfaults
                work(0xBADADD)

            counter += 1
        return counter

    r = foo()
    print(r)

In the terminal (``...`` on a line by itself indicates output that is not
presented for brevity), note the setting of ``NUMBA_NUM_THREADS`` to 4 to ensure
that there are 4 threads running in the parallel section:

.. code-block:: none
    :emphasize-lines: 1, 19, 29, 44, 50, 56, 62, 69

    $ NUMBA_NUM_THREADS=4 NUMBA_OPT=0 python demo_gdb_threads.py
    Attaching to PID: 21462
    ...
    Attaching to process 21462
    [New LWP 21467]
    [New LWP 21468]
    [New LWP 21469]
    [New LWP 21470]
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    0x00007f59ec31756d in nanosleep () at ../sysdeps/unix/syscall-template.S:81
    81      T_PSEUDO (SYSCALL_SYMBOL, SYSCALL_NAME, SYSCALL_NARGS)
    Breakpoint 1 at 0x7f59d631e8f0: file numba/_helperlib.c, line 1090.
    Continuing.
    [Switching to Thread 0x7f59d1fd1700 (LWP 21470)]

    Thread 5 "python" hit Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    (gdb) info threads
    Id   Target Id         Frame
    1    Thread 0x7f59eca2f740 (LWP 21462) "python" pthread_cond_wait@@GLIBC_2.3.2 ()
        at ../nptl/sysdeps/unix/sysv/linux/x86_64/pthread_cond_wait.S:185
    2    Thread 0x7f59d37d4700 (LWP 21467) "python" pthread_cond_wait@@GLIBC_2.3.2 ()
        at ../nptl/sysdeps/unix/sysv/linux/x86_64/pthread_cond_wait.S:185
    3    Thread 0x7f59d2fd3700 (LWP 21468) "python" pthread_cond_wait@@GLIBC_2.3.2 ()
        at ../nptl/sysdeps/unix/sysv/linux/x86_64/pthread_cond_wait.S:185
    4    Thread 0x7f59d27d2700 (LWP 21469) "python" numba_gdb_breakpoint () at numba/_helperlib.c:1090
    * 5    Thread 0x7f59d1fd1700 (LWP 21470) "python" numba_gdb_breakpoint () at numba/_helperlib.c:1090
    (gdb) thread apply 2-5 info locals

    Thread 2 (Thread 0x7f59d37d4700 (LWP 21467)):
    No locals.

    Thread 3 (Thread 0x7f59d2fd3700 (LWP 21468)):
    No locals.

    Thread 4 (Thread 0x7f59d27d2700 (LWP 21469)):
    No locals.

    Thread 5 (Thread 0x7f59d1fd1700 (LWP 21470)):
    sched$35 = '\000' <repeats 55 times>
    counter__arr = '\000' <repeats 16 times>, "\001\000\000\000\000\000\000\000\b\000\000\000\000\000\000\000\370B]\"hU\000\000\001", '\000' <repeats 14 times>
    counter = 0
    (gdb) continue
    Continuing.
    [Switching to Thread 0x7f59d27d2700 (LWP 21469)]

    Thread 4 "python" hit Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    (gdb) continue
    Continuing.
    [Switching to Thread 0x7f59d1fd1700 (LWP 21470)]

    Thread 5 "python" hit Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    (gdb) continue
    Continuing.
    [Switching to Thread 0x7f59d27d2700 (LWP 21469)]

    Thread 4 "python" hit Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    (gdb) continue
    Continuing.

    Thread 5 "python" received signal SIGSEGV, Segmentation fault.
    [Switching to Thread 0x7f59d1fd1700 (LWP 21470)]
    __GI___libc_free (mem=0xbadadd) at malloc.c:2935
    2935      if (chunk_is_mmapped(p))                       /* release mmapped memory. */
    (gdb) bt
    #0  __GI___libc_free (mem=0xbadadd) at malloc.c:2935
    #1  0x00007f59d37ded84 in $3cdynamic$3e::__numba_parfor_gufunc__0x7ffff80a61ae3e31$244(Array<unsigned long long, 1, C, mutable, aligned>, Array<long long, 1, C, mutable, aligned>) () at <string>:24
    #2  0x00007f59d17ce326 in __gufunc__._ZN13$3cdynamic$3e45__numba_parfor_gufunc__0x7ffff80a61ae3e31$244E5ArrayIyLi1E1C7mutable7alignedE5ArrayIxLi1E1C7mutable7alignedE ()
    #3  0x00007f59d37d7320 in thread_worker ()
    from <path>/numba/numba/npyufunc/workqueue.cpython-37m-x86_64-linux-gnu.so
    #4  0x00007f59ec626e25 in start_thread (arg=0x7f59d1fd1700) at pthread_create.c:308
    #5  0x00007f59ec350bad in clone () at ../sysdeps/unix/sysv/linux/x86_64/clone.S:113

In the output it can be seen that there are 4 threads launched and that they all
break at the breakpoint, further that ``Thread 5`` receives a signal ``SIGSEGV``
and that back tracing shows that it came from ``__GI___libc_free`` with the
invalid address in ``mem``, as expected.

Using the ``gdb`` command language
----------------------------------
Both the :func:`numba.gdb` and :func:`numba.gdb_init` functions accept unlimited
string arguments which will be passed directly to ``gdb`` as command line
arguments when it initializes, this makes it easy to set breakpoints on other
functions and perform repeated debugging tasks without having to manually type
them every time. For example, this code runs with ``gdb`` attached and sets a
breakpoint on ``_dgesdd`` (say for example the arguments passed to the LAPACK's
double precision divide and conqueror SVD function need debugging).

.. code-block:: python
  :linenos:

    from numba import njit, gdb
    import numpy as np

    @njit(debug=True)
    def foo(a):
        # instruct Numba to attach gdb at this location and on launch, switch
        # breakpoint pending on , and then set a breakpoint on the function
        # _dgesdd, continue execution, and once the breakpoint is hit, backtrace
        gdb('-ex', 'set breakpoint pending on',
            '-ex', 'b dgesdd_',
            '-ex','c',
            '-ex','bt')
        b = a + 10
        u, s, vh = np.linalg.svd(b)
        return s # just return singular values

    z = np.arange(70.).reshape(10, 7)
    r = foo(z)
    print(r)

In the terminal (``...`` on a line by itself indicates output that is not
presented for brevity), note that no interaction is required to break and
backtrace:

.. code-block:: none
    :emphasize-lines: 1

    $ NUMBA_OPT=0 python demo_gdb_args.py
    Attaching to PID: 22300
    GNU gdb (GDB) Red Hat Enterprise Linux 8.0.1-36.el7
    ...
    Attaching to process 22300
    Reading symbols from <py_env>/bin/python3.7...done.
    0x00007f652305a550 in __nanosleep_nocancel () at ../sysdeps/unix/syscall-template.S:81
    81      T_PSEUDO (SYSCALL_SYMBOL, SYSCALL_NAME, SYSCALL_NARGS)
    Breakpoint 1 at 0x7f650d0618f0: file numba/_helperlib.c, line 1090.
    Continuing.

    Breakpoint 1, numba_gdb_breakpoint () at numba/_helperlib.c:1090
    1090    }
    Breakpoint 2 at 0x7f65102322e0 (2 locations)
    Continuing.

    Breakpoint 2, 0x00007f65182be5f0 in mkl_lapack.dgesdd_ ()
    from <py_env>/lib/python3.7/site-packages/numpy/core/../../../../libmkl_rt.so
    #0  0x00007f65182be5f0 in mkl_lapack.dgesdd_ ()
    from <py_env>/lib/python3.7/site-packages/numpy/core/../../../../libmkl_rt.so
    #1  0x00007f650d065b71 in numba_raw_rgesdd (kind=kind@entry=100 'd', jobz=<optimized out>, jobz@entry=65 'A', m=m@entry=10,
        n=n@entry=7, a=a@entry=0x561c6fbb20c0, lda=lda@entry=10, s=0x561c6facf3a0, u=0x561c6fb680e0, ldu=10, vt=0x561c6fd375c0,
        ldvt=7, work=0x7fff4c926c30, lwork=-1, iwork=0x7fff4c926c40, info=0x7fff4c926c20) at numba/_lapack.c:1277
    #2  0x00007f650d06768f in numba_ez_rgesdd (ldvt=7, vt=0x561c6fd375c0, ldu=10, u=0x561c6fb680e0, s=0x561c6facf3a0, lda=10,
        a=0x561c6fbb20c0, n=7, m=10, jobz=65 'A', kind=<optimized out>) at numba/_lapack.c:1307
    #3  numba_ez_gesdd (kind=<optimized out>, jobz=<optimized out>, m=10, n=7, a=0x561c6fbb20c0, lda=10, s=0x561c6facf3a0,
        u=0x561c6fb680e0, ldu=10, vt=0x561c6fd375c0, ldvt=7) at numba/_lapack.c:1477
    #4  0x00007f650a3147a3 in numba::targets::linalg::svd_impl::$3clocals$3e::svd_impl$243(Array<double, 2, C, mutable, aligned>, omitted$28default$3d1$29) ()
    #5  0x00007f650a1c0489 in __main__::foo$241(Array<double, 2, C, mutable, aligned>) () at demo_gdb_args.py:15
    #6  0x00007f650a1c2110 in cpython::__main__::foo$241(Array<double, 2, C, mutable, aligned>) ()
    #7  0x00007f650cd096a4 in call_cfunc ()
    from <path>/numba/numba/_dispatcher.cpython-37m-x86_64-linux-gnu.so
    ...


How does the ``gdb`` binding work?
----------------------------------
For advanced users and debuggers of Numba applications it's important to know
some of the internal implementation details of the outlined ``gdb`` bindings.
The :func:`numba.gdb` and :func:`numba.gdb_init` functions work by injecting the
following into the function's LLVM IR:

* At the call site of the function first inject a call to ``getpid(3)`` to get
  the PID of the executing process and store this for use later, then inject a
  ``fork(3)`` call:

  * In the parent:

    * Inject a call ``sleep(3)`` (hence the pause whilst ``gdb`` loads).
    * Inject a call to the ``numba_gdb_breakpoint`` function (only
      :func:`numba.gdb` does this).

  * In the child:

    * Inject a call to ``execl(3)`` with the arguments
      ``numba.config.GDB_BINARY``, the ``attach`` command and the PID recorded
      earlier. Numba has a special ``gdb`` command file that contains
      instructions to break on the symbol ``numba_gdb_breakpoint`` and then
      ``finish``, this is to make sure that the program stops on the
      breakpoint but the frame it stops in is the compiled Python frame (or
      one ``step`` away from, depending on optimisation). This command file is
      also added to the arguments and finally and any user specified arguments
      are added.

At the call site of a :func:`numba.gdb_breakpoint` a call is injected to the
special ``numba_gdb_breakpoint`` symbol, which is already registered and
instrumented as a place to break and ``finish`` immediately.

As a result of this, a e.g. :func:`numba.gdb` call will cause a fork in the
program, the parent will sleep whilst the child launches ``gdb`` and attaches it
to the parent and tells the parent to continue. The launched ``gdb`` has the
``numba_gdb_breakpoint`` symbol registered as a breakpoint and when the parent
continues and stops sleeping it will immediately call ``numba_gdb_breakpoint``
on which the child will break. Additional :func:`numba.gdb_breakpoint` calls
create calls to the registered breakpoint hence the program will also break at
these locations.


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
