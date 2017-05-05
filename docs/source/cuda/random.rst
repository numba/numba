
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

Example
'''''''

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
