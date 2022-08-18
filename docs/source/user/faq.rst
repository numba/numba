
==========================
Frequently Asked Questions
==========================

Installation
============

Numba could not be imported
---------------------------

If you are seeing an exception on importing Numba with an error message
that starts with::

    ImportError: Numba could not be imported.

here are some common issues and things to try to fix it.

#. Your installation has more than one version of Numba a given environment.

   Common ways this occurs include:

   * Installing Numba with conda and then installing again with pip.
   * Installing Numba with pip and then updating to a new version with pip (pip
     re-installations don't seem to always clean up very well).

   To fix this the best approach is to create an entirely new environment and
   install a single version of Numba in that environment using a package manager
   of your choice.

#. Your installation has Numba for Python version X but you are running with
   Python version Y.

   This occurs due to a variety of Python environment mix-up/mismatch problems.
   The most common mismatch comes from installing Numba into the
   site-packages/environment of one version of Python by using a base or
   system installation of Python that is a different version, this typically
   happens through the use of the "wrong" ``pip`` binary. This will obviously
   cause problems as the C-Extensions on which Numba relies are bound to
   specific Python versions. A way to check if this likely the problem is to
   see if the path to the ``python`` binary at::

       python -c 'import sys; print(sys.executable)'

   matches the path to your installation tool and/or matches the reported
   installation location and if the Python versions match up across all of
   these. Note that Python version ``X.Y.A`` is compatible with ``X.Y.B``.

   To fix this the best approach is to create an entirely new environment and
   ensure that the installation tool used to install Numba is the one from that
   environment/the Python versions at install and run time match.

#. Your core system libraries are too old.

   This is a somewhat rare occurrence, but there are occasions when a very old
   (typically out of support) version of Linux is in use it doesn't have a
   ``glibc`` library with sufficiently new versioned symbols for Numba's shared
   libraries to resolve against. The fix for this is to update your OS system
   libraries/update your OS.

#. You are using an IDE e.g. Spyder.

   There are some unknown issues in relation to installing Numba via IDEs, but
   it would appear that these are likely variations of 1. or 2. with the same
   suggested fixes. Also, try installation from outside of the IDE with the
   command line.


If you have an installation problem which is not one of the above problems,
please do ask on `numba.discourse.group <https://numba.discourse.group/>`_ and
if possible include the path where Numba is installed and also the output of::

    python -c 'import sys; print(sys.executable)'


Programming
===========

Can I pass a function as an argument to a jitted function?
----------------------------------------------------------

As of Numba 0.39, you can, so long as the function argument has also been
JIT-compiled::

   @jit(nopython=True)
   def f(g, x):
       return g(x) + g(-x)

   result = f(jitted_g_function, 1)

However, dispatching with arguments that are functions has extra overhead.
If this matters for your application, you can also use a factory function to 
capture the function argument in a closure::

   def make_f(g):
       # Note: a new f() is created each time make_f() is called!
       @jit(nopython=True)
       def f(x):
           return g(x) + g(-x)
       return f

   f = make_f(jitted_g_function)
   result = f(1)

Improving the dispatch performance of functions in Numba is an ongoing task.

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

If the ``parallel=True`` transformations failed for a function
decorated as such, a warning will be displayed. See also
:ref:`numba-parallel-diagnostics` for information about parallel diagnostics.

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

Why has my loop not vectorized?
-------------------------------

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


.. note::
   Using ``--debug-only`` requires LLVM to be build with assertions enabled to
   work. Use the build of llvmlite in the `Numba channel <https://anaconda.org/numba/llvmlite>`_
   which is linked against LLVM with assertions enabled.

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

Why are the ``typed`` containers slower when used from the interpreter?
-----------------------------------------------------------------------

The Numba ``typed`` containers found in ``numba.typed`` e.g.
``numba.typed.List`` store their data in an efficient form for access from JIT
compiled code. When these containers are used from the CPython interpreter, the
data involved has to be converted from/to the container format. This process is
relatively costly and as a result impacts performance. In JIT compiled code no
such penalty exists and so operations on the containers are much quicker and
often faster than the pure Python equivalent.

Does Numba automatically parallelize code?
------------------------------------------

It can, in some cases:

* Ufuncs and gufuncs with the ``target="parallel"`` option will run on multiple threads.
* The ``parallel=True`` option to ``@jit`` will attempt to optimize array
  operations and run them in parallel.  It also adds support for ``prange()`` to
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


GPU Programming
===============

How do I work around the ``CUDA initialized before forking`` error?
-------------------------------------------------------------------

On Linux, the ``multiprocessing`` module in the Python standard library
defaults to using the ``fork`` method for creating new processes.  Because of
the way process forking duplicates state between the parent and child
processes, CUDA will not work correctly in the child process if the CUDA
runtime was initialized *prior* to the fork.  Numba detects this and raises a
``CudaDriverError`` with the message ``CUDA initialized before forking``.

One approach to avoid this error is to make all calls to ``numba.cuda``
functions inside the child processes or after the process pool is created.
However, this is not always possible, as you might want to query the number of
available GPUs before starting the process pool.  In Python 3, you can change
the process start method, as described in the `multiprocessing documentation
<https://docs.python.org/3.9/library/multiprocessing.html#contexts-and-start-methods>`_.
Switching from ``fork`` to ``spawn`` or ``forkserver`` will avoid the CUDA
initialization issue, although the child processes will not inherit any global
variables from their parent.


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

How do I get Numba development builds?
--------------------------------------

Pre-release versions of Numba can be installed with conda::

    $ conda install -c numba/label/dev numba


Miscellaneous
=============

Where does the project name "Numba" come from?
----------------------------------------------

"Numba" is a combination of "NumPy" and "Mamba". Mambas are some of the fastest
snakes in the world, and Numba makes your Python code fast.

How do I reference/cite/acknowledge Numba in other work?
--------------------------------------------------------
For academic use, the best option is to cite our ACM Proceedings: `Numba: a
LLVM-based Python JIT compiler.
<http://dl.acm.org/citation.cfm?id=2833162&dl=ACM&coll=DL>`_ You can also find
`the sources on github <https://github.com/numba/Numba-SC15-Paper>`_, including
`a pre-print pdf
<https://github.com/numba/Numba-SC15-Paper/raw/master/numba_sc15.pdf>`_, in case
you don't have access to the ACM site but would like to read the paper.

Other related papers
~~~~~~~~~~~~~~~~~~~~
A paper describing ParallelAccelerator technology, that is activated when the
``parallel=True`` jit option is used, can be found `here
<http://drops.dagstuhl.de/opus/volltexte/2017/7269/pdf/LIPIcs-ECOOP-2017-4.pdf>`_.

How do I write a minimal working reproducer for a problem with Numba?
---------------------------------------------------------------------

A minimal working reproducer for Numba should include:

1. The source code of the function(s) that reproduce the problem.
2. Some example data and a demonstration of calling the reproducing code with
   that data. As Numba compiles based on type information, unless your problem
   is numerical, it's fine to just provide dummy data of the right type, e.g.
   use ``numpy.ones`` of the correct ``dtype``/size/shape for arrays.
3. Ideally put 1. and 2. into a script with all the correct imports. Make sure
   your script actually executes and reproduces the problem before submitting
   it! The target is to make it so that the script can just be copied directly
   from the `issue tracker <https://github.com/numba/numba/issues>`_ and run by
   someone else such that they can see the same problem as you are having.

Having made a reproducer, now remove every part of the code that does not
contribute directly to reproducing the problem to create a "minimal" reproducer.
This means removing imports that aren't used, removing variables that aren't
used or have no effect, removing lines of code which have no effect, reducing
the complexity of expressions, and shrinking input data to the minimal amount
required to trigger the problem.

Doing the above really helps out the Numba issue triage process and will enable
a faster response to your problem!

`Suggested further reading
<http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_ on
writing minimal working reproducers.
