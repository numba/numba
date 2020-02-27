.. _numba-threading-layer:

The Threading Layers
====================

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

If you installed Numba with ``pip``, TBB can be enabled by running::

    $ pip install tbb

Due to compatibility issues with manylinux1 and other portability concerns,
the OpenMP threading layer is disabled in the Numba binary wheels on PyPI.

.. note::
    The default manner in which Numba searches for and loads a threading layer
    is tolerant of missing libraries, incompatible runtimes etc.


.. _numba-threading-layer-setting-mech:

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
Parallel execution is fundamentally derived from core Python libraries in four
forms (the first three also apply to code using parallel execution via other
means!):

* ``threads`` from the ``threading`` module.
* ``spawn`` ing processes from the ``multiprocessing`` module via ``spawn``
  (default on Windows, only available in Python 3.4+ on Unix)
* ``fork`` ing processes from the ``multiprocessing`` module via ``fork``
  (default on Unix).
* ``fork`` ing processes from the ``multiprocessing`` module through the use of
  a ``forkserver`` (only available in Python 3 on Unix). Essentially a new
  process is spawned and then forks are made from this new process on request.

Any library in use with these forms of parallelism must exhibit safe behaviour
under the given paradigm. As a result, the threading layer selection methods
are designed to provide a way to choose a threading layer library that is safe
for a given paradigm in an easy, cross platform and environment tolerant manner.
The options that can be supplied to the
:ref:`setting mechanisms <numba-threading-layer-setting-mech>` are as
follows:

* ``default`` provides no specific safety guarantee and is the default.
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

    # this will force the compilation of the function, select a threading layer
    # and then execute in parallel
    foo(x, y)

    # demonstrate the threading layer chosen
    print("Threading layer chosen: %s" % threading_layer())

which produces::

    Threading layer chosen: omp

and this makes sense as GNU OpenMP, as present on Linux, is thread safe.

Selecting a named threading layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Advanced users may wish to select a specific threading layer for their use case,
this is done by directly supplying the threading layer name to the
:ref:`setting mechanisms <numba-threading-layer-setting-mech>`. The options
and requirements are as follows:

+----------------------+-----------+-------------------------------------------+
| Threading Layer Name | Platform  | Requirements                              |
+======================+===========+===========================================+
| ``tbb``              | All       | The ``tbb`` package (``$ conda install    |
|                      |           | tbb``)                                    |
+----------------------+-----------+-------------------------------------------+
| ``omp``              | Linux     | GNU OpenMP libraries (very likely this    |
|                      |           | will already exist)                       |
|                      |           |                                           |
|                      | Windows   | MS OpenMP libraries (very likely this will|
|                      |           | already exist)                            |
|                      |           |                                           |
|                      | OSX       | The ``intel-openmp`` package (``$ conda   |
|                      |           | install intel-openmp``)                   |
+----------------------+-----------+-------------------------------------------+
| ``workqueue``        | All       | None                                      |
+----------------------+-----------+-------------------------------------------+

Should the threading layer not load correctly Numba will detect this and provide
a hint about how to resolve the problem. It should also be noted that the Numba
diagnostic command ``numba -s`` has a section
``__Threading Layer Information__`` that reports on the availability of
threading layers in the current environment.


Extra notes
-----------
The threading layers have fairly complex interactions with CPython internals and
system level libraries, some additional things to note:

* The installation of Intel's TBB libraries vastly widens the options available
  in the threading layer selection process.
* On Linux, the ``omp`` threading layer is not fork safe due to the GNU OpenMP
  runtime library (``libgomp``) not being fork safe. If a fork occurs in a
  program that is using the ``omp`` threading layer, a detection mechanism is
  present that will try and gracefully terminate the forked child and print an
  error message to ``STDERR``.
* On OSX, the ``intel-openmp`` package is required to enable the OpenMP based
  threading layer.

.. _setting_the_number_of_threads:

Setting the Number of Threads
-----------------------------

The number of threads used by numba is based on the number of CPU cores
available (see :obj:`numba.config.NUMBA_DEFAULT_NUM_THREADS`), but it can be
overridden with the :envvar:`NUMBA_NUM_THREADS` environment variable.

The total number of threads that numba launches is in the variable
:obj:`numba.config.NUMBA_NUM_THREADS`.

For some use cases, it may be desirable to set the number of threads to a
lower value, so that numba can be used with higher level parallelism.

The number of threads can be set dynamically at runtime using
:func:`numba.set_num_threads`. Note that :func:`~.set_num_threads` only allows
setting the number of threads to a smaller value than
:obj:`~.NUMBA_NUM_THREADS`. Numba always launches
:obj:`numba.config.NUMBA_NUM_THREADS` threads, but :func:`~.set_num_threads`
causes it to mask out unused threads so they aren't used in computations.

The current number of threads used by numba can be accessed with
:func:`numba.get_num_threads`. Both functions work inside of a jitted
function.

.. _numba-threading-layer-thread-masking:

Example of Limiting the Number of Threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, suppose the machine we are running on has 8 cores (so
:obj:`numba.config.NUMBA_NUM_THREADS` would be ``8``). Suppose we want to run
some code with ``@njit(parallel=True)``, but we also want to run our code
concurrently in 4 different processes. With the default number of threads,
each Python process would run 8 threads, for a total in 4*8 = 32 threads,
which is oversubscription for our 8 cores. We should rather limit each process
to 2 threads, so that the total will be 4*2 = 8, which matches our number of
physical cores.

There are two ways to do this. One is to set the :envvar:`NUMBA_NUM_THREADS`
environment variable to ``2``.

.. code:: bash

   $ NUMBA_NUM_THREADS=2 python ourcode.py

However, there are two downsides to this approach:

1. :envvar:`NUMBA_NUM_THREADS` must be set before Numba is imported, and
   ideally before Python is launched. As soon as Numba is imported the
   environment variable is read and that number of threads is locked in as the
   number of threads Numba launches.

2. If we want to later increase the number of threads used by the process, we
   cannot. :envvar:`NUMBA_NUM_THREADS` sets the *maximum* number of threads
   that are launched for a process. Calling :func:`~.set_num_threads()` with a
   value greater than :obj:`numba.config.NUMBA_NUM_THREADS` results in an
   error.

The advantage of this approach is that we can do it from outside of the
process without changing the code.

Another approach is to use the :func:`numba.set_num_threads` function in our code

.. code:: python

   from numba import njit, set_num_threads

   @njit(parallel=True)
   def func():
       ...

   set_num_threads(2)
   func()

If we call ``set_num_threads(2)`` before executing our parallel code, it has
the same effect as calling the process with ``NUMBA_NUM_THREADS=2``, in that
the parallel code will only execute on 2 threads. However, we can later call
``set_num_threads(8)`` to increase the number of threads back to the default
size. And we do not have to worry about setting it before Numba gets imported.
It only needs to be called before the parallel function is run.

API Reference
~~~~~~~~~~~~~

.. py:data:: numba.config.NUMBA_NUM_THREADS

   The total (maximum) number of threads launched by numba.

   Defaults to :obj:`numba.config.NUMBA_DEFAULT_NUM_THREADS`, but can be
   overridden with the :envvar:`NUMBA_NUM_THREADS` environment variable.

.. py:data:: numba.config.NUMBA_DEFAULT_NUM_THREADS

   The number of CPU cores on the system (as determined by
   ``multiprocessing.cpu_count()``). This is the default value for
   :obj:`numba.config.NUMBA_NUM_THREADS` unless the
   :envvar:`NUMBA_NUM_THREADS` environment variable is set.

.. autofunction:: numba.set_num_threads

.. autofunction:: numba.get_num_threads
