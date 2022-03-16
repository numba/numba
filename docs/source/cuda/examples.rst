
========
Examples
========

.. _cuda-vecadd:

Vector Addition
=====================
This example shows a very basic vector addition using Numba to create both the
device based data arrays and the vector addition kernel. It serves as a good warmup
for learning the bare bones of what is needed to write GPU kernels using numba.


.. code-block:: python

   from numba import cuda
   import numpy as np

This function represents the kernel. Note the function is defined in terms of 
python variables with unknown types. Later, when launched, numba will examine the
types of the arguments that are actually passed at runtime and use them to 
generate a CUDA kernel for the correct primitive types. The ``size`` parameter 
is used as an out of bounds thread guard. 

Also note that just like CUDA kernels in ``c`` are declared as ``void``, and must 
return their values through an array that is passed, Numba kernels also do 
not return values. Here, let ``c`` represent our results.

.. code-block:: python

   @cuda.jit
   def f(a, b, c, size):
      # just like threadIdx.x + (blockIdx.x * blockDim.x)
      tid = cuda.grid(1) 
      
      if tid < size:
         c[tid] = a[tid] + b[tid]

``numba.cuda.to_device`` can be used create device side copies of numpy arrays.
Create two data vectors and an empty vector to hold our results:

.. code-block:: python

   a = cuda.to_device(np.array([1,2,3]))
   b = cuda.to_device(np.array([1,2,3]))
   c = cuda.to_device(np.array([0,0,0]))


The following call to ``forall`` autoconfigures a 1D kernel for the data size
and is often the simplest way of launching a kernel.

.. code-block:: python

   f.forall(len(a))(a, b, c, len(a))
   print(c.copy_to_host())

This prints

.. code-block:: none

   [2 4 6]


One can also configure the grid manually using the following syntax to launch
a grid containing one block with three threads:

.. code-block:: python

   f[1, 3](a, b, c, len(a))
   print(c.copy_to_host())

This also prints

.. code-block:: none

   [2 4 6]

.. _cuda-laplace:

1D Heat Equation
=====================
This example solves Laplace's equation in one dimension for a certain set of initial
conditions and boundary conditions. A full discussion of Laplace's equation is out of
scope for this documentation, but it will suffice to say that it describes how heat
propagates through an object over time. It works by discretizing the problem in two ways:

1. The object is separated into small "pieces" that each have an individual temperature
2. Time is separated into small "intervals" and the universe advances through them one by one

Then, the following assumption is applied: The temperature of a piece after some interval 
has passed is some weighted average of the temperature of the pieces that are directly 
touching it. Intuitively, this means if all the small pieces of the object are very hot 
and a single piece in the middle is very cold, as time goes on, the hot pieces will cause 
the cold one to heat up and the cold piece will cause the surrounding hot pieces to cool 
slightly. Simply put, the heat spreads out around the object.

We can simulate this situation using a Kernel generated from Numba. Let's start simple by assuming
we have a one dimensional object which we'll represent with an array of values. The position 
of the element in the array is like the position of a small piece of the object, and the value 
of the element is like the temperature. 

.. code-block:: python

   from numba import cuda
   import numpy as np


Some initial setup here. Let's make one point in the center of the object very hot.

.. code-block:: python

   size = 100
   data = np.zeros(size)

   # make one piece in the very middle very hot
   data[int(size/2)] = 10000
   data_gpu = cuda.to_device(data)


In our kernel each thread will be responsible for managing the temperature update for a single element
in a loop over the desired number of timesteps. The kernel is below:

.. code-block:: python

   @cuda.jit
   def solve_heat_equation(data, size, timesteps, k):
      i = cuda.grid(1)

      for step in range(timesteps):
         # get the current temperature associated with this segment
         curr_temp = data[i]
         
         # apply formula from finite difference equation
         if i == 0:
               # Left wall is held at T = 0
               next_temp = curr_temp + k * (data[i + 1] - (2 * curr_temp))
         elif i == size - 1:
               # Right wall is held at T = 0
               next_temp = curr_temp + k * (data[i - 1] - (2 * curr_temp))
         else:
               next_temp = curr_temp + k * (data[i - 1] - (2 * curr_temp) + data[i + 1])
         data[i] = next_temp
         
         # wait for every thread to write before moving on
         cuda.syncthreads()

Calling the kernel:

.. code-block:: python

   solve_heat_equation.forall(len(data))(data_gpu, len(data_gpu), 10000, 0.25)

Since the kernel reused the original data vector to work on the problem, copying it back
to the host yields the results:

.. code-block:: python

   print(data_gpu.copy_to_host())

This prints

.. code-block:: none

   array([ 0.54632855,  1.09212753,  1.6368679 ,  2.18002166,  2.72106236,
         3.25946564,  3.79470973,  4.32627596,  4.85364931,  5.37631887,
         5.89377839,  6.40552673,  6.91106841,  7.409914  ,  7.90158064,
         8.38559243,  8.86148082,  9.32878491,  9.78705184, 10.23583701,
         10.67470447, 11.10322728, 11.52098821, 11.92758065, 12.32260997,
         12.7056948 , 13.07646632, 13.43456455, 13.7796327 , 14.11129136,
         14.42908512, 14.73343975, 15.0248911 , 15.30232574, 15.56474082,
         15.81201588, 16.04395606, 16.26034615, 16.46097692, 16.64565132,
         16.81418871, 16.9664262 , 17.10221779, 17.22143367, 17.32395972,
         17.40969746, 17.47856409, 17.53049271, 17.56543251, 17.58334895,
         17.58422393, 17.56805586, 17.53485974, 17.48466713, 17.41752618,
         17.33350153, 17.23267429, 17.1151419 , 16.98101808, 16.83043268,
         16.66353154, 16.48047636, 16.2814445 , 16.06662887, 15.83623766,
         15.5904942 , 15.32963669, 15.053918  , 14.76360541, 14.45898035,
         14.14033813, 13.80798763, 13.46225106, 13.10346357, 12.73197298,
         12.34813942, 11.95233499, 11.5449434 , 11.12635958, 10.69698932,
         10.25724886,  9.80756451,  9.3483722 ,  8.8801171 ,  8.40325314,
         7.91824262,  7.42555573,  6.9256701 ,  6.41907034,  5.90624758,
         5.38769897,  4.86392722,  4.33544009,  3.80274994,  3.26637319,
         2.72682981,  2.18464289,  1.64033805,  1.09444297,  0.54748687])

Plotting this data with a graphing library shows an arc shape highest where
the object was hot initially and gradually sloping down to zero towards the
edges where the temperature is fixed at zero. In the limit of infinite time,
the arc will flatten out completely.

.. _cuda_reduction_shared:

Shared Memory Reduction
=======================
Numba exposes many CUDA features including shared memory. Shared memory is high speed on chip memory
that represents a shared address space that all the threads in a block can see. It is local to each
block and is limited in size, but extremely quick to access, similar to a cache. More can be read 
about shared memory in the CUDA documentation. As a way of demonstrating shared memory, let's reimplement 
a famous CUDA solution for summing a vector which works by "folding" the data up using a sucessively
smaller number of threads.

.. code-block:: python

   from numba import cuda
   import numpy as np
   from numba.types import int32

   # generate data
   a = cuda.to_device(np.arange(1024))
   nelem = len(a)

Here is a version of the kernel implemented using Numba:

.. code-block:: python

   @cuda.jit
   def array_sum(data, size):
      tid = cuda.threadIdx.x
      if tid < size:
         i = cuda.grid(1)
         
         # declare an array in shared memory
         shr = cuda.shared.array(nelem, int32)
         shr[tid] = data[i]
         
         # make sure every thread has written its value to shared memory
         # before we start reducing
         cuda.syncthreads()
         
         s = 1
         while s < cuda.blockDim.x:
            if tid % (2 * s) == 0:
                  # stride by `s` and add
                  shr[tid] += shr[tid + s]
            s *= 2
            cuda.syncthreads()
            
         # after the loop, the zeroth element contains the sum
         if tid == 0:
            data[tid] = shr[tid]

This kernel can be run as follows and the same result is obtained through
summing the host data by traditional means.

.. code-block:: python

   kernel.forall(len(a))(a, len(a))
   print(a[0]) # array(523776)
   sum(np.arange(1024)) # 523776

This algorithm can be greatly improved upon by redesigning the inner loop
to use sequential memory accesses, or even furthermore using strategies that
keep more threads active and working since in this example most threads quickly
become idle.

.. _cuda_sessionization:

Dividing Click Data into Sessions
=================================


A common problem in business analytics is that of grouping the activity of users of an online platform into
sessions, called "sessionization". The idea is that users generally traverse through a website and perform
various actions (clicking something, filling out a form, etc) in discrete groups. Perhaps a customer spends
some time shopping for an item in the morning and then again at night - often the business is interested in
treating these periods as separate interactions with their service, and this creates the problem of 
programmatically splitting up activity in some agreed upon way.

Here we'll illustrate how to write a group of numba kernels to solve this problem. We'll start with data 
containing two fields: let user_id represent a unique ID corresponding to an individual customer, and let 
action_time be a time that some unknown action was taken on the service. Right now, we'll assume there's 
only one type of action, so all there is to know is when it happened.

Our goal will be to create a new column called session_id which contains a label corresponding to a unique 
session. We'll define the boundary between sessions as when there has been at least one hour between clicks.


.. code-block:: python

   from numba import cuda
   import numpy as np

   # set the timeout to one hour
   session_timeout = np.int64(np.timedelta64('3600', 's'))
   
Here is a solution using Numba:

.. code-block:: python

   @cuda.jit
   def sessionize(
      user_id, timestamp, results, size
   ):
      gid = cuda.grid(1)
      
      
      ### DETERMINE SESSION BOUNDARIES
      is_sess_boundary = 0
      if 0 < gid < size:
         if (
               user_id[gid] != user_id[gid - 1]
               or timestamp[gid] - timestamp[gid - 1] > session_timeout
         ):
               is_sess_boundary = 1
         else:
               is_sess_boundary = 0
      else:
         is_sess_boundary = 1
         
         
      ### DETERMINE SESSION LABELS
      # this thread marks the start of a session
      if gid < size:
         if is_sess_boundary:
               results[gid] = gid
               look_ahead = 1

               # check elements 'forward' of this one 
               # until a new session boundary is found
               while results[gid + look_ahead] == 0:
                  results[gid + look_ahead] = gid
                  look_ahead += 1

                  # don't segfault if I'm the last thread
                  if gid + look_ahead == size - 1:
                     results[gid + look_ahead] = gid
                     break

Let's generate some data and try out the kernel.

.. code-block:: python

   # generate data
   ids = cuda.to_device(
      np.array([1,1,1,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4])
   )

   sec = cuda.to_device(
      np.array(
         [1,2,3,5000,5001,5002,1,2,3,1,2,5000,5001,10000,10001,10002,10003,15000,150001,1,5000,50001,15000,20000,25000,25001,25002, 25003],
         dtype='datetime64[ns]').astype('int64') # cast to int64 for compatibility
   )

   # create a vector to hold the results
   results = cuda.to_device(np.zeros(len(ids)))

   # kernel call
   sessionize.forall(len(ids))(ids, sec, results, len(ids))
   
   print(results.copy_to_host())
   # array([ 0.,  0.,  0.,  3.,  3.,  3.,  6.,  6.,  6.,  9.,  9., 11., 11.,
   #    13., 13., 13., 13., 17., 18., 19., 20., 21., 21., 23., 24., 24.,
   #    24., 24.])


As can be seen above, the kernel successfully divided the first three datapoints from the second three for the first user ID,
and a similar pattern is seen througout. This example has wide applicability in a business setting and can be used to study
many problems around consumer behavior and product interaction.

.. _cuda_reuse_function:

JIT Function CPU-GPU Compatibility
==================================

This example demonstrates how ``numba.jit`` can be used to jit compile a function for the CPU, while at the same time making 
it available for use inside CUDA kernels. This can be very useful for users that are migrating workflows from CPU to GPU as 
they can directly reuse potential business logic with fewer code changes.

Take the following example function:

.. code-block:: python

   from math import pi

   @numba.jit
   def business_logic(x, y, z):
      return 4 * z * (2 * x - (4 * y) / 2 * pi)

The function ``business_logic`` can be run standalone in compiled form on the CPU:

..code-block:: python

   print(business_logic(1,2,3)) # -126.79644737231007

It can also be directly reused threadwise inside a GPU kernel. For example one may 
generate some vectors to represent ``x``, ``y``, and ``z``:

.. code-block:: python

   X = cp.array([1, 10, 234])
   Y = cp.array([2,2,4014])
   Z = cp.array([3,14,2211])

   results = cp.array([0.,0.,0.])

And a numba kernel referencing the decorated function:

.. code-block:: python

   @cuda.jit
   def f(res, xarr, yarr, zarr, size):
      tid = cuda.grid(1)
      
      if tid < size:
         # the function decorated with numba.jit can just be directly reused
         res[tid] = business_logic(xarr[tid], yarr[tid], zarr[tid])

This kernel can be invoked in the normal way:

.. code-block:: python

   f.forall(len(X))(results, X, Y, Z, len(X))
   print(results) # [-126.79644737231007, 416.28324559588634, -218912930.2987788]

.. _cuda_montecarlo:

Monte Carlo Integration
=======================

This example shows how to use numba to approximate the value of a definite integral by rapidly generating 
random numbers on the GPU. A detailed description of the mathematical mechanics of monte carlo integeration 
is out of the scope of the example, but it can briefly be described as an averaging process where the area 
under the curve is approximated by taking the average of many rectangles formed by its function values.

.. code-block:: python

   nsamps = 1000000 # numer of samples, higher will lead to a more accurate answer

Here's a kernel and a convenience function that calls it:

.. code-block:: python

   @cuda.jit
   def mc_integrator_kernel(out, rng_states, lower_lim, upper_lim, size):
      """
      kernel to draw random samples and evaluate the function to
      be integrated at those sample values
      """
      
      gid = cuda.grid(1)
      if (gid < size):
         # draw a sample between 0 and 1 on this thread
         samp = xoroshiro128p_uniform_float32(rng_states, gid)
         
         # normalize this sample to the limit range
         samp = samp * (upper_lim - lower_lim) + lower_lim
         
         # evaluate the function to be integrated at the normalized
         # value of the sample
         y = func(samp)
         out[gid] = y

      
   def mc_integrate(lower_lim, upper_lim, nsamps):
      """
      approximate the definite integral of `func` from
      `lower_lim` to `upper_lim`
      """
      out = cp.zeros(nsamps, dtype='float32')
      rng_states = create_xoroshiro128p_states(nsamps, seed=42)
      
      # jit the function for use in CUDA kernels
      
      mc_integrator_kernel.forall(nsamps)(out, rng_states, lower_lim, upper_lim, nsamps)

      # normalization factor to convert to the average: (b - a)/(N - 1)
      factor = (upper_lim - lower_lim) / (nsamps - 1)
      
      return out.sum() * factor

We can now use ``mc_integrate`` to compute the definite integral of this function between
two limits:

.. code-block:: python

   mc_integrate(1, 2, 1000000) # array(0.6929643, dtype=float32)
   mc_integrate(2, 3, 1000000) # array(0.4054021, dtype=float32)

.. _cuda-matmul:

Matrix multiplication
=====================
First, import the modules needed for this example:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
   :language: python
   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
   :start-after: magictoken.ex_import.begin
   :end-before: magictoken.ex_import.end
   :dedent: 8
   :linenos:

Here is a naïve implementation of matrix multiplication using a CUDA kernel:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
   :language: python
   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
   :start-after: magictoken.ex_matmul.begin
   :end-before: magictoken.ex_matmul.end
   :dedent: 8
   :linenos:

An example usage of this function is as follows:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
   :language: python
   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
   :start-after: magictoken.ex_run_matmul.begin
   :end-before: magictoken.ex_run_matmul.end
   :dedent: 8
   :linenos:

This implementation is straightforward and intuitive but performs poorly,
because the same matrix elements will be loaded multiple times from device
memory, which is slow (some devices may have transparent data caches, but
they may not be large enough to hold the entire inputs at once).

It will be faster if we use a blocked algorithm to reduce accesses to the
device memory.  CUDA provides a fast :ref:`shared memory <cuda-shared-memory>`
for threads in a block to cooperatively compute on a task.  The following
implements a faster version of the square matrix multiplication using shared
memory:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
   :language: python
   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
   :start-after: magictoken.ex_fast_matmul.begin
   :end-before: magictoken.ex_fast_matmul.end
   :dedent: 8
   :linenos:


Because the shared memory is a limited resource, the code preloads a small
block at a time from the input arrays.  Then, it calls
:func:`~numba.cuda.syncthreads` to wait until all threads have finished
preloading and before doing the computation on the shared memory.
It synchronizes again after the computation to ensure all threads
have finished with the data in shared memory before overwriting it
in the next loop iteration.

An example usage of the ``fast_matmul`` function is as follows:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
   :language: python
   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
   :start-after: magictoken.ex_run_fast_matmul.begin
   :end-before: magictoken.ex_run_fast_matmul.end
   :dedent: 8
   :linenos:


This passes a :ref:`CUDA memory check test <debugging-cuda-python-code>`, which
can help with debugging. Running the code above produces the following output:

.. code-block:: none

    $ python fast_matmul.py
    [[ 6.  6.  6.  6.]
    [22. 22. 22. 22.]
    [38. 38. 38. 38.]
    [54. 54. 54. 54.]]
    [[ 6.  6.  6.  6.]
    [22. 22. 22. 22.]
    [38. 38. 38. 38.]
    [54. 54. 54. 54.]]

.. note:: For high performance matrix multiplication in CUDA, see also the `CuPy implementation <https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html>`_.

The approach outlined here generalizes to non-square matrix multiplication as
follows by adjusting the ``blockspergrid`` variable:

Again, here is an example usage:

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
   :language: python
   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
   :start-after: magictoken.ex_run_nonsquare.begin
   :end-before: magictoken.ex_run_nonsquare.end
   :dedent: 8
   :linenos:

and the corresponding output:

.. code-block:: none

  $ python nonsquare_matmul.py
  [[ 253.  253.  253.  253.  253.  253.  253.]
  [ 782.  782.  782.  782.  782.  782.  782.]
  [1311. 1311. 1311. 1311. 1311. 1311. 1311.]
  [1840. 1840. 1840. 1840. 1840. 1840. 1840.]
  [2369. 2369. 2369. 2369. 2369. 2369. 2369.]]
  [[ 253.  253.  253.  253.  253.  253.  253.]
  [ 782.  782.  782.  782.  782.  782.  782.]
  [1311. 1311. 1311. 1311. 1311. 1311. 1311.]
  [1840. 1840. 1840. 1840. 1840. 1840. 1840.]
  [2369. 2369. 2369. 2369. 2369. 2369. 2369.]]
