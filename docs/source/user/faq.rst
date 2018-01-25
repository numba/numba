
==========================
Frequently Asked Questions
==========================


Programming
===========

Can I pass a function as an argument to a jitted function?
----------------------------------------------------------

You can't, but in many cases you can use a closure to emulate it.
For example, this example::

   @jit(nopython=True)
   def f(g, x):
       return g(x) + g(-x)

   result = f(my_g_function, 1)

could be rewritten using a factory function::

   def make_f(g):
       # Note: a new f() is compiled each time make_f() is called!
       @jit(nopython=True)
       def f(x):
           return g(x) + g(-x)
       return f

   f = make_f(my_g_function)
   result = f(1)

Numba doesn't seem to care when I modify a global variable
----------------------------------------------------------

Numba considers global variables as compile-time constants.  If you want
your jitted function to update itself when you have modified a global
variable's value, one solution is to recompile it using the
:meth:`~Dispatcher.recompile` method.  This is a relatively slow operation,
though, so you may instead decide to rearchitect your code and turn the
global variable into a function argument.

Can I debug a jitted function?
------------------------------

Calling into :mod:`pdb` or other such high-level facilities is currently not
supported from Numba-compiled code.  However, you can temporarily disable
compilation by setting the :envvar:`NUMBA_DISABLE_JIT` environment
variable.

How can I create a Fortran-ordered array?
-----------------------------------------

Numba currently doesn't support the ``order`` argument to most Numpy
functions such as :func:`numpy.empty` (because of limitations in the
:term:`type inference` algorithm).  You can work around this issue by
creating a C-ordered array and then transposing it.  For example::

   a = np.empty((3, 5), order='F')
   b = np.zeros(some_shape, order='F')

can be rewritten as::

   a = np.empty((5, 3)).T
   b = np.zeros(some_shape[::-1]).T

How can I increase integer width?
---------------------------------

By default, Numba will generally use machine integer width for integer
variables.  On a 32-bit machine, you may sometimes need the magnitude of
64-bit integers instead.  You can simply initialize relevant variables as
``np.int64`` (for example ``np.int64(0)`` instead of ``0``).  It will
propagate to all computations involving those variables.

.. _parallel_faqs:

How can I tell if ``parallel=True`` worked?
-------------------------------------------

Set the :ref:`environment variable <numba-envvars>` ``NUMBA_WARNINGS`` to
non-zero and if the ``parallel=True`` transformations failed for a function
decorated as such, a warning will be displayed.

Also, setting the :ref:`environment variable <numba-envvars>`
``NUMBA_DEBUG_ARRAY_OPT_STATS`` will show some statistics about which
operators/calls are converted to parallel for-loops.

Performance
===========

Does Numba inline functions?
----------------------------

Numba gives enough information to LLVM so that functions short enough
can be inlined.  This only works in :term:`nopython mode`.

Does Numba vectorize array computations (SIMD)?
-----------------------------------------------

Numba doesn't implement such optimizations by itself, but it lets LLVM
apply them.

Why my loop is not vectorized?
------------------------------

Numba enables the loop-vectorize optimization in LLVM by default.
While it is a powerful optimization, not all loops are applicable.
Sometimes, loop-vectorization may fail due to subtle details like memory access
pattern. To see additional diagnostic information from LLVM,
add the following lines:

.. code-block:: python

    import llvmlite.binding as llvm
    llvm.set_option('', '--debug-only=loop-vectorize')

This tells LLVM to print debug information from the **loop-vectorize**
pass to stderr.  Each function entry looks like:

.. code-block:: text

    LV: Checking a loop in "<low-level symbol name>" from <function name>
    LV: Loop hints: force=? width=0 unroll=0
    ...
    LV: Vectorization is possible but not beneficial.
    LV: Interleaving is not beneficial.

Each function entry is separated by an empty line.  The reason for rejecting
the vectorization is usually at the end of the entry.  In the example above,
LLVM rejected the vectorization because doing so will not speedup the loop.
In this case, it can be due to memory access pattern.  For instance, the
array being looped over may not be in contiguous layout.

When memory access pattern is non-trivial such that it cannot determine the
access memory region, LLVM may reject with the following message:

.. code-block:: text

    LV: Can't vectorize due to memory conflicts

Another common reason is:

.. code-block:: text

    LV: Not vectorizing: loop did not meet vectorization requirements.

In this case, vectorization is rejected because the vectorized code may behave
differently.  This is a case to try turning on ``fastmath=True`` to allow
fastmath instructions.


Does Numba automatically parallelize code?
------------------------------------------

It can, in some cases:

* Ufuncs and gufuncs with the ``target="parallel"`` option will run on multiple threads.
* The experimental ``parallel=True`` option to ``@jit`` will attempt to optimize
  array operations and run them in parallel.  It also adds support for ``prange()`` to
  explicitly parallelize a loop.

You can also manually run computations on multiple threads yourself and use
the ``nogil=True`` option (see :ref:`releasing the GIL <jit-nogil>`).  Numba
can also target parallel execution on GPU architectures using its CUDA and HSA
backends.


Can Numba speed up short-running functions?
-------------------------------------------

Not significantly.  New users sometimes expect to JIT-compile such
functions::

   def f(x, y):
       return x + y

and get a significant speedup over the Python interpreter.  But there isn't
much Numba can improve here: most of the time is probably spent in CPython's
function call mechanism, rather than the function itself.  As a rule of
thumb, if a function takes less than 10 µs to execute: leave it.

The exception is that you *should* JIT-compile that function if it is called
from another jitted function.

There is a delay when JIT-compiling a complicated function, how can I improve it?
---------------------------------------------------------------------------------

Try to pass ``cache=True`` to the ``@jit`` decorator.  It will keep the
compiled version on disk for later use.

A more radical alternative is :ref:`ahead-of-time compilation <pycc>`.


Integration with other utilities
================================

Can I "freeze" an application which uses Numba?
-----------------------------------------------

If you're using PyInstaller or a similar utility to freeze an application,
you may encounter issues with llvmlite.  llvmlite needs a non-Python DLL
for its working, but it won't be automatically detected by freezing utilities.
You have to inform the freezing utility of the DLL's location: it will
usually be named ``llvmlite/binding/libllvmlite.so`` or
``llvmlite/binding/llvmlite.dll``, depending on your system.

I get errors when running a script twice under Spyder
-----------------------------------------------------

When you run a script in a console under Spyder, Spyder first tries to
reload existing modules.  This doesn't work well with Numba, and can
produce errors like ``TypeError: No matching definition for argument type(s)``.

There is a fix in the Spyder preferences. Open the "Preferences" window,
select "Console", then "Advanced Settings", click the "Set UMR excluded
modules" button, and add ``numba`` inside the text box that pops up.

To see the setting take effect, be sure to restart the IPython console or
kernel.

.. _llvm-locale-bug:

Why does Numba complain about the current locale?
-------------------------------------------------

If you get an error message such as the following::

   RuntimeError: Failed at nopython (nopython mode backend)
   LLVM will produce incorrect floating-point code in the current locale

it means you have hit a LLVM bug which causes incorrect handling of
floating-point constants.  This is known to happen with certain third-party
libraries such as the Qt backend to matplotlib.

To work around the bug, you need to force back the locale to its default
value, for example::

   import locale
   locale.setlocale(locale.LC_NUMERIC, 'C')


Miscellaneous
=============

How do I reference/cite/acknowledge Numba in other work?
--------------------------------------------------------
For academic use, the best option is to cite our ACM Proceedings:
`Numba: a LLVM-based Python JIT compiler.
<http://dl.acm.org/citation.cfm?id=2833162&dl=ACM&coll=DL>`_
