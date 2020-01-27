=========================================
Notes on Numba's threading implementation
=========================================

The execution of the work presented by the Numba ``parallel`` targets is
undertaken by the Numba threading layer. Practically, the "threading layer"
is a Numba built-in library that can perform the required concurrent execution.
At the time of writing there are three threading layers available, each
implemented via a different lower level native threading library. More
information on the threading layers and appropriate selection of a threading
layer for a given application/system can be found in the
:ref:`threading layer documentation <numba-threading-layer>`.

The pertinent information to note for the following sections is that the
function in the threading library that performs the parallel execution is the
``parallel_for`` function. The job of this function is to both orchestrate and
execute the parallel tasks.

The relevant source files referenced in this document are

- ``numba/npyufunc/tbbpool.cpp``
- ``numba/npyufunc/omppool.cpp``
- ``numba/npyufunc/workqueue.c``

  These files contain the TBB, OpenMP, and workqueue threadpool
  implementations, respectively. Each includes the functions
  ``set_num_threads()``, ``get_num_threads()``, and ``get_thread_id()``, as
  well as the relevant logic for thread masking in their respective
  schedulers. Note that the basic thread local variable logic is duplicated in
  each of these files, and not shared between them.

- ``numba/npyufunc/parallel.py``

  This file contains the Python and JIT compatible wrappers for
  ``set_num_threads()``, ``get_num_threads()``, and ``get_thread_id()``, as
  well as the code that loads the above libraries into Python and launches the
  threadpool.

- ``numba/npyufunc/parfor.py``

  This file contains the main logic for generating code for the parallel
  backend. The thread mask is accessed in this file in the code that generates
  scheduler code, and passed to the relevant backend scheduler function (see
  below).

Thread masking
--------------

As part of its design, Numba never launches new threads beyond the threads
that are launched initially with ``numba.npyufunc.parallel._launch_threads()``
when the first parallel execution is run. This is due to the way threads were
already implemented in Numba prior to thread masking being implemented. This
restriction was kept to keep the design simple, although it could be removed
in the future. Consequently, it's possible to programmatically set the number
of threads, but only to less than or equal to the total number that have
already been launched. This is done by "masking" out unused threads, causing
them to do no work. For example, on a 16 core machine, if the user were to
call ``set_num_threads(4)``, Numba would always have 16 threads present, but
12 of them would sit idle for parallel computations. A further call to
``set_num_threads(16)`` would cause those same threads to do work in later
computations.

:ref:`Thread masking <numba-threading-layer-thread-masking>` was added to make
it possible for a user to programmatically alter the number of threads
performing work in the threading layer. Thread masking proved challenging to
implement as it required the development of a programming model that is suitable
for users, easy to reason about, and could be implemented safely, with
consistent behavior across the various threading layers.

Programming model
~~~~~~~~~~~~~~~~~

The programming model chosen is similar to that found in OpenMP. The reasons
for this choice were that it is familiar to a lot of users, restricted in
scope and also simple. The number of threads in use is specified by calling
``set_num_threads`` and the number of threads in use can be queried by calling
``get_num_threads``.These two functions are synonymous with their OpenMP
counterparts (with the above restriction that the mask must be less than or
equal to the number of launched threads). The execution semantics are also
similar to OpenMP in that once a parallel region is launched, altering the
thread mask has no impact on the currently executing region, but will have an
impact on parallel regions executed subsequently.

The Implementation
~~~~~~~~~~~~~~~~~~

So as to place no further restrictions on user code other than those that
already existed in the threading layer libraries, careful consideration of the
design of thread masking was required. The "thread mask" cannot be stored in a
global value as concurrent use of the threading layer may result in classic
forms of race conditions on the value itself. Numerous designs were discussed
involving various types of mutex on such a global value, all of which were
eventually broken through thought experiment alone. It eventually transpired
that, following some OpenMP implementations, the "thread mask" is best
implemented as a ``thread local``. This means each thread that executes a Numba
parallel function will have a thread local storage (TLS) slot that contains the
value of the thread mask to use when scheduling threads in the ``parallel_for``
function.

The above notion of TLS use for a thread mask is relatively easy to implement,
``get_num_threads`` and ``set_num_threads`` simply need to address the TLS slot
in a given threading layer. This also means that the execution schedule for a
parallel region can be derived from a run time call to ``get_num_threads``. This
is achieved via a well known and relatively easy to implement pattern of a ``C``
library function registration and wrapping it in the internal Numba
implementation.

In addition to satisfying the original upfront thread masking requirements, a
few more complicated scenarios needed consideration as follows.

Nested parallelism
******************

In all threading layers a "main thread" will invoke the ``parallel_for``
function and then in the parallel region, depending on the threading layer,
some number of additional threads will assist in doing the actual work.
If the work contains a call to another parallel function (i.e. nested
parallelism) it is necessary for the thread making the call to know what the
"thread mask" of the main thread is so that it can propagate it into the
``parallel_for`` call it makes when executing the nested parallel function.
The implementation of this behavior is threading layer specific but the general
principle is for the "main thread" to always "send" the value of the thread mask
from its TLS slot to all threads in the threading layer that are active in the
parallel region. These active threads then update their TLS slots with this
value prior to performing any work. The net result of this implementation detail
is that:

* thread masks correctly propagate into nested functions
* it's still possible for each thread in a parallel region to safely have a
  different mask with which to call nested functions, if it's not set explicitly
  then the inherited mask from the "main thread" is used
* threading layers which have dynamic scheduling with threads potentially
  joining and leaving the active pool during a ``parallel_for`` execution are
  successfully accommodated
* any "main thread" thread mask is entirely decoupled from the in-flux nature
  of the thread masks of the threads in the active thread pool

Python threads independently invoking parallel functions
********************************************************

The threading layer launch sequence is heavily guarded to ensure that the
launch is both thread and process safe and run once per process. In a system
with numerous Python ``threading`` module threads all using Numba, the first
thread through the launch sequence will get its thread mask set appropriately,
but no further threads can run the launch sequence. This means that other
threads will need their initial thread mask set some other way. This is
achieved when ``get_num_threads`` is called and no thread mask is present, in
this case the thread mask will be set to the default. In the implementation,
"no thread mask is present" is represented by the value ``-1`` and the "default
thread mask" (unset) is represented by the value ``0``. The implementation also
immediately calls ``set_num_threads(NUMBA_NUM_THREADS)`` after doing this, so
if either ``-1`` or ``0`` is encountered as a result from ``get_num_threads()`` it
indicates a bug in the above processes.

OS ``fork()`` calls
*******************

The use of TLS was also in part driven by the Linux (the most popular
platform for Numba use by far) having a ``fork(2, 3P)`` call that will do TLS
propagation into child processes, see ``clone(2)``\ 's ``CLONE_SETTLS``.

Thread ID
*********

A private ``get_thread_id()`` function was added to each threading backend,
which returns a unique ID for each thread. This can be accessed from Python by
``numba.npyufunc.parallel._get_thread_id()`` (it can also be used inside a
JIT compiled function). The thread ID function is useful for testing that the
thread masking behavior is correct, but it should not be used outside of the
tests. For example, one can call ``set_num_threads(4)`` and then collect all
unique ``_get_thread_id()``\ s in a parallel region to verify that only 4
threads are run.

Caveats
~~~~~~~

Some caveats to be aware of when testing thread masking:

- The TBB backend may choose to schedule fewer than the given mask number of
  threads. Thus a test such as the one described above may return fewer than 4
  unique threads.

- The workqueue backend is not threadsafe, so attempts to do multithreading
  nested parallelism with it may result in deadlocks or other undefined
  behavior. The workqueue backend will raise a SIGABRT signal if it detects
  nested parallelism.

- Certain backends may reuse the main thread for computation, but this
  behavior shouldn't be relied upon (for instance, if propagating exceptions).

Use in Code Generation
~~~~~~~~~~~~~~~~~~~~~~

The general pattern for using ``get_num_threads`` in code generation is

.. code:: python

   import llvmlite.llvmpy.core as lc

   get_num_threads = builder.module.get_or_insert_function(
       lc.Type.function(lc.Type.int(types.intp.bitwidth), []),
       name="get_num_threads")

   num_threads = builder.call(get_num_threads, [])

   with cgutils.if_unlikely(builder, builder.icmp_signed('<=', num_threads,
                                                 num_threads.type(0))):
       cgutils.printf(builder, "num_threads: %d\n", num_threads)
       context.call_conv.return_user_exc(builder, RuntimeError,
                                                 ("Invalid number of threads. "
                                                  "This likely indicates a bug in Numba.",))

   # Pass num_threads through to the appropriate backend function here

See the code in ``numba/npyufunc/parfor.py``.

The guard against ``num_threads`` being <= 0 is not strictly necessary, but it
can protect against accidentally incorrect behavior in case the thread masking
logic contains a bug.

The ``num_threads`` variable should be passed through to the appropriate
backend function, such as ``do_scheduling`` or ``parallel_for``. If it's used
in some way other than passing it through to the backend function, the above
considerations should be taken into account to ensure the use of the
``num_threads`` variable is safe. It would probably be better to keep such
logic in the threading backends, rather than trying to do it in code
generation.
