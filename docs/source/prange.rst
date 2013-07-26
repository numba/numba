Parallel Range
==============

Numba implements the ability to run loops in parallel, similar to OpenMP parallel
for loops and Cython's ``prange``. The loops body is scheduled in seperate threads,
and they execute in a ``nopython`` numba context. ``prange`` automatically
takes care of data privatization and reductions::

    from numba import autojit, prange

    @autojit
    def parallel_sum(A):
        sum = 0.0
        for i in prange(A.shape[0]):
            sum += A[i]

        return sum

Here the variable ``sum`` is a reduction variable that is automatically summed at the end
of the parallel loop.

Privatization rules are simple, in order of importance:

    * Variables that are operated on with inplace operators are reduction variables.
    * Variables that are assigned to are private to each thead

        * The variable will assume the sequentially last value after the loop.
          This is the equivalent of OpenMP's lastprivate clause.

    * Variables that are only read are shared between all threads

The order above specifies that reductions trump privates, and privates trump shared
variables::

    @autojit
    def privatization_rules():
        reduction = 1.0
        private = 2.0
        shared = 3.0
        for i in prange(100):
            reduction += i      # The inplace operator specifies a sum reduction
            reduction -= 1
            reduction *= 4      # ERROR: inconsistent reduction operator!
                                # '*' is a product reduction, not a sum reduction

            print private       # ERROR: private is not yet initialized!
            private = i * 4.0   # This assignment makes it private
            print private       # Private is available now, this is fine

            print shared        # This variable is only ever read, so it's shared

        print reduction         # prints the sum-reduced value
        print private           # prints the last value, i.e. 99 * 4.0

.. NOTE:: Although ``prange`` introduces a ``nopython`` context, it does not actually
          release the GIL. In addition to not being able to use objects, it is invalid
          to try to obtain the GIL in the ``prange`` body or a function called from there.
          This will result in a deadlock.

Currently ``prange`` will use as many CPUs as detected by the ``multiprocessing`` module.
It is likely that in the next release it will accept a ``num_threads`` clause to allow
this to be parameterized. ``prange`` may also accept a ``schedule`` clause in the future
to allow specifying how iterations should be scheduled.