.. _numba-threading-layer:

The Numba threading layer
=========================

This section is about the Numba threading layer, this is the library that is
used internally to perform the parallel execution that occurs through the use of
the ``parallel`` targets for CPUs, namely:

* The use of the ``parallel=True`` kwarg in ``@jit`` and ``@njit``.
* The use of the ``target='parallel'`` kwarg in ``@vectorize`` and
  ``@guvectorize``.

.. note::
    If a code base does not use the ``threading`` or ``multiprocessing``
    modules (or any other sort of parallelism) the defaults for the threading
    layer that ship with Numba will work well, no further action is required!


Which threading layers are available?
-------------------------------------
There are three threading layers available and they are named as follows:

* ``tbb`` - A threading layer backed by Intel TBB.
* ``omp`` - A threading layer backed by OpenMP.
* ``workqueue`` -A simple built-in work-sharing task scheduler.

In practice, the only threading layer guaranteed to be present is ``workqueue``.
The ``omp`` layer requires the presence of a suitable OpenMP runtime library.
The ``tbb`` layer requires the presence of Intel's TBB libraries, these can be
obtained via the conda command::

    $ conda install tbb

.. note::
    The default manner in which Numba searches for and loads a threading layer
    is tolerant to missing libraries, incompatible runtimes etc.


Setting the threading layer
---------------------------
The threading layer is set via the environment variable
``NUMBA_THREADING_LAYER`` or through assignment to
``numba.config.THREADING_LAYER``. If the programmatic approach to setting the
threading layer is used it must occur logically before any Numba based
compilation for a parallel target has occurred. There are two approaches to
choosing a threading layer, the first is by selecting a threading layer that is
safe under various forms of parallel execution, the second is through explicit
selection via the threading layer name (e.g. ``tbb``).

Selecting a threading layer for safe parallel execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parallel execution is fundamentally derived from core Python libraries in three
forms (this also applies to code using parallel execution via other means!):

* ``threads`` from the ``threading`` module.
* ``spawn`` ing processes from the ``multiprocessing`` module via ``spawn``
  (not available on Unix for Python 2.x, also Python 3.x ``multiprocessing``
  ``forkserver`` also ``spawn`` s).
* ``fork`` ing processes from the ``multiprocessing`` module via ``fork``
  (Unix only).

Any library in use with these forms of parallelism must exhibit safe behaviour
under the given paradigm. As a result, the threading layer selection methods
are designed to provide a way to choose a threading layer library that is safe
for a given paradigm in an easy, cross platform and environment tolerant manner.
The options are as follows:

* ``default`` provides no safety and is the default.
* ``safe`` is both fork and thread safe, this requires the ``tbb`` package
  (Intel TBB libraries) to be installed.
* ``forksafe`` provides a fork safe library.
* ``threadsafe`` provides a thread safe library.

To discover the threading layer that was selected, the function
``numba.threading_layer()`` may be called after parallel execution. For example,
on a Linux machine with no TBB installed::

    from numba import config, njit, threading_layer
    import numpy as np

    # set the threading layer before any parallel target compilation
    config.THREADING_LAYER = 'threadsafe'

    @njit(parallel=True)
    def foo(a, b):
        return a + b

    x = np.arange(10.)
    y = x.copy()

    # this will force the compilation of the function, select a threading layer and
    # then execute in parallel
    foo(x, y)

    # demonstrate the threading layer chosen
    print("Threading layer chosen: %s" % threading_layer())

which produces::

    Threading layer chosen: omp

and this makes sense as GNU OpenMP, as present on Linux, is thread safe.

Extra notes
-----------
The threading layers have fairly complex interactions with CPython internals and
system level libraries, some additional things to note:

* The installation of Intel TBB vastly widens the options available in the
  threading layer selection process.
* For Linux and Python 2.7, ``Process`` based parallelism from the
  ``multiprocessing`` module will be achieve through the use of ``fork`` ing.
* For Linux, the ``omp`` threading layer is not fork safe due to the GNU OpenMP
  runtime library (``libgomp``) not being fork safe. If a fork occurs in a
  program that is using the ``omp`` threading layer, a detection mechanism is
  present that will try and gracefully terminate the forked child and print an
  error message to ``STDERR``.
