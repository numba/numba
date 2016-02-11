
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

Does Numba parallelize code?
----------------------------

No, it doesn't.  If you want to run computations concurrently on multiple
threads (by :ref:`releasing the GIL <jit-nogil>`) or processes, you'll
have to handle the pooling and synchronisation yourself.

Or, you can take a look at NumbaPro_.

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


.. _NumbaPro: http://docs.continuum.io/numbapro/
