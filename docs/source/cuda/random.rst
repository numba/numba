
.. _cuda-random:

Random Number Generation
========================

Numba provides a random number generation algorithm that can be executed on
the GPU.  Due to technical issues with how NVIDIA implemented cuRAND, however,
Numba's GPU random number generator is not based on cuRAND.  Instead, Numba's
GPU RNG is an implementation of the `xoroshiro128+ algorithm
<http://xoroshiro.di.unimi.it/>`_. The xoroshiro128+ algorithm has a period of
``2**128 - 1``, which is shorter than the period of the XORWOW algorithm
used by default in cuRAND, but xoroshiro128+ still passes the BigCrush tests
of random number generator quality.

When using any RNG on the GPU, it is important to make sure that each thread
has its own RNG state, and they have been initialized to produce non-overlapping
sequences.  The  numba.cuda.random module provides a host function to do this,
as well as CUDA device functions to obtain uniformly or normally distributed
random numbers.

.. note:: Numba (like cuRAND) uses the
    `Box-Muller transform <https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>`
    to generate normally distributed random numbers from a uniform generator.
    However, Box-Muller generates pairs of random numbers, and the current
    implementation only returns one of them.  As a result, generating normally
    distributed values is half the speed of uniformly distributed values.

.. automodule:: numba.cuda.random
    :members: create_xoroshiro128p_states, init_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64, xoroshiro128p_normal_float32, xoroshiro128p_normal_float64
    :noindex:

A simple example
''''''''''''''''

Here is a sample program that uses the random number generator::

    from __future__ import print_function, absolute_import

    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
    import numpy as np

    @cuda.jit
    def compute_pi(rng_states, iterations, out):
        """Find the maximum value in values and store in result[0]"""
        thread_id = cuda.grid(1)

        # Compute pi by drawing random (x, y) points and finding what
        # fraction lie inside a unit circle
        inside = 0
        for i in range(iterations):
            x = xoroshiro128p_uniform_float32(rng_states, thread_id)
            y = xoroshiro128p_uniform_float32(rng_states, thread_id)
            if x**2 + y**2 <= 1.0:
                inside += 1

        out[thread_id] = 4.0 * inside / iterations

    threads_per_block = 64
    blocks = 24
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    out = np.zeros(threads_per_block * blocks, dtype=np.float32)

    compute_pi[blocks, threads_per_block](rng_states, 10000, out)
    print('pi:', out.mean())

An example of managing RNG state size and using a 3D grid
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The number of RNG states scales with the number of threads using the RNG, so it
is often better to use strided loops in conjunction with the RNG in order to
keep the state size manageable.

In the following example, which initializes a large 3D array with random
numbers, using one thread per output element would result in 453,617,100 RNG
states.  This would take a long time to initialize and poorly utilize the GPU.
Instead, it uses a fixed size 3D grid with a total of 2,097,152 (``(16 ** 3) *
(8 ** 3)``) threads striding over the output array. The 3D thread indices
``startx``, ``starty``, and ``startz``  are linearized into a 1D index,
``tid``, to index into the 2,097,152 RNG states.


.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_random.py
   :language: python
   :caption: from ``test_ex_3d_grid of ``numba/cuda/tests/doc_example/test_random.py``
   :start-after: magictoken.ex_3d_grid.begin
   :end-before: magictoken.ex_3d_grid.end
   :dedent: 8
   :linenos:
