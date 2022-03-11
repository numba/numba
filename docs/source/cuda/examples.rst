
========
Examples
========

.. _cuda-vecadd:

Vector Addition
=====================
This example shows a very basic vector addition using Numba to create both the
device based data arrays and the vector addition kernel.


.. code-block:: python
   from numba import cuda
   import numpy as np

This function represents the kernel. Note the function is defined in terms of 
python variables with unknown types. Later, when launched, numba will examine the
types of the arguments that are actually passed at runtime and use them to 
generate a CUDA kernel for the correct primitive types. The `size` parameter 
is used as an out of bounds thread guard. 

Also note that just like CUDA kernels that are declared as `void`, and must 
return their values through an array that is passed, Numba kernels also do 
not return values. Here, let `c` represent our results.

.. code-block:: python
   @cuda.jit
   def f(a, b, c, size):
      # just like threadIdx.x + (blockIdx.x * blockDim.x)
      tid = cuda.grid(1) 
      
      if tid < size:
         c[tid] = a[tid] + b[tid]

`numba.cuda.to_device` can be used create device side copies of numpy arrays.
Create two data vectors and an empty vector to hold our results:

.. code-block:: python
   a = cuda.to_device(np.array([1,2,3]))
   b = cuda.to_device(np.array([1,2,3]))
   c = cuda.to_device(np.array([0,0,0]))


The following call to `forall` autoconfigures a 1D kernel for the data size
and is often the simplest way of launching a kernel.

..code-block:: python
   f.forall(
      len(a)
   )(
      a, b, c, len(a)
   )
   print(c.copy_to_host())

This prints

..code-block:: none
   [2 4 6]

One can also configure the grid manually using the following syntax to launch
a grid containing one block with three threads:

..code-block:: python
   f[1, 3](a, b, c, len(a))
   print(c.copy_to_host())

This also prints
..code-block:: none
   [2 4 6]


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
