
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

Thread masking
--------------
:ref:`Thread masking <numba-threading-layer-thread-masking>` was added to make
it possible for a user to programmatically alter the number of threads
performing work in the threading layer. Thread masking proved challenging to
implement as it required the development of a programming model that is suitable
for users, easy to reason about, and could be implemented safely, with
consistent behaviour across the various threading layers.

Programming model
~~~~~~~~~~~~~~~~~
The programming model chosen is similar to that found in OpenMP. The reasons for
this choice were that it is familiar to a lot of users, restricted in scope and
also simple. The number of threads in use is specified by calling
``set_num_threads`` and the number of threads in use can be queried by calling
``get_num_threads``.These two functions are synonymous with their OpenMP
counterparts. The execution semantic is also similar to OpenmP in that once a
parallel region is launched altering the thread mask has no impact on the
currently executing region but will have an impact on parallel regions executed
subsequently.


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
The implementation of this behaviour is threading layer specific but the general
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
thread through the launch sequence will get its thread mask set
appropriately, but no further threads can run the launch sequence. This means
that other threads will need their initial thread mask set some other way,
this is achieved when ``get_num_threads`` is called and no thread mask is
present, in this case the thread mask will be set to the default.

OS ``fork()`` calls
*******************

The use of TLS was also in part driven by the Linux (the most popular
platform for Numba use by far) having a ``fork(2, 3P)`` call that will do TLS
propagation into child processes, see ``clone(2)``'s ``CLONE_SETTLS``.

