from __future__ import print_function, absolute_import
import math

from numba import cuda, float32, float64, int64, uint64, void, from_dtype

import numpy as np

# This implementation is based upon the xoroshiro128+ and splitmix64 algorithms
# described at:
#
#     http://xoroshiro.di.unimi.it/
#
# and originally implemented by David Blackman and Sebastiano Vigna.
#
# The implementations below are based on the C source code:
#
#  * http://xoroshiro.di.unimi.it/xoroshiro128plus.c
#  * http://xoroshiro.di.unimi.it/splitmix64.c
#
# Splitmix64 is used to generate the initial state of the xoroshiro128+ generator
# to ensure that small seeds don't result in predictable output.


xoroshiro128p_dtype = np.dtype([('s0', np.uint64), ('s1', np.uint64)])
xoroshiro128p_type = from_dtype(xoroshiro128p_dtype)


@cuda.jit(void(xoroshiro128p_type[:], int64, uint64), device=True)
def init_xoroshiro128p_state(states, index, seed):
    '''Use SplitMix64 to generate an xoroshiro128p state from 64-bit seed.

    This ensures that manually set small seeds don't result in a predictable
    initial sequence from the random number generator 

    Parameters
    ----------
    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: uint64
    :param index: offset in states to update
    :type seed: int64
    :param seed: seed value to use when initializing state
    '''
    z = seed + uint64(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)) * uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * uint64(0x94D049BB133111EB)
    z = z ^ (z >> 31)

    states[index]['s0'] = z
    states[index]['s1'] = z


@cuda.jit('uint64(uint64, int32)', device=True)
def rotl(x, k):
    '''Left rotate x by k bits.'''
    return (x << k) | (x >> (64 - k))


@cuda.jit(uint64(xoroshiro128p_type[:], int64), device=True)
def xoroshiro128p_next(states, index):
    '''Return the next random uint64 and advance the RNG in states[index].

    Parameters
    ----------
    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update

    '''
    s0 = states[index]['s0']
    s1 = states[index]['s1']
    result = s0 + s1
    
    s1 ^= s0
    states[index]['s0'] = rotl(s0, 55) ^ s1 ^ (s1 << 14)
    states[index]['s1'] = rotl(s1, 36)
    
    return result


XOROSHIRO128P_JUMP = (uint64(0xbeac0467eba5facb), uint64(0xd86b048b86aa9922))


@cuda.jit(void(xoroshiro128p_type[:], int64), device=True)
def xoroshiro128p_jump(states, index):
    '''Advance the RNG in states[index] by 2**64 steps.

    Parameters
    ----------
    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update

    '''
    s0 = uint64(0)
    s1 = uint64(0)
    
    for i in range(2):
        for b in range(64):
            if XOROSHIRO128P_JUMP[i] & uint64(1) << b:
                s0 ^= states[index]['s0']
                s1 ^= states[index]['s1']
            xoroshiro128p_next(states, index)
    
    states[index]['s0'] = s0
    states[index]['s1'] = s1


@cuda.jit('float64(uint64)', device=True)
def uint64_to_unit_float64(x):
    '''Convert uint64 to float64 value in the range [0.0, 1.0)'''
    return (x >> 11) * (float64(1) / (uint64(1) << 53))


@cuda.jit('float32(uint64)', device=True)
def uint64_to_unit_float32(x):
    '''Convert uint64 to float64 value in the range [0.0, 1.0)'''
    return float32(uint64_to_unit_float64(x))


@cuda.jit(float32(xoroshiro128p_type[:], int64), device=True)
def xoroshiro128p_uniform_float32(states, index):
    '''Return a float32 in range [0.0, 1.0) and advance states[index].

    Parameters
    ----------
    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update

    '''
    return uint64_to_unit_float32(xoroshiro128p_next(states, index))


@cuda.jit(float64(xoroshiro128p_type[:], int64), device=True)
def xoroshiro128p_uniform_float64(states, index):
    '''Return a float64 in range [0.0, 1.0) and advance states[index].

    Parameters
    ----------
    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update

    '''
    return uint64_to_unit_float64(xoroshiro128p_next(states, index))


TWO_PI_FLOAT32 = np.float32(2 * math.pi)
TWO_PI_FLOAT64 = np.float64(2 * math.pi)

@cuda.jit(float32(xoroshiro128p_type[:], int64), device=True)
def xoroshiro128p_normal_float32(states, index):
    '''Return a normally distributed float32 and advance states[index].

    The return value is drawn from a Gaussian of mean=0 and sigma=1 using the
    Box-Muller transform.  This advances the RNG sequence by two steps.

    Parameters
    ----------
    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update

    '''
    u1 = xoroshiro128p_uniform_float32(states, index)
    u2 = xoroshiro128p_uniform_float32(states, index)

    z0 = math.sqrt(-float32(2.0) * math.log(u1)) * math.cos(TWO_PI_FLOAT32 * u2)
    # discarding second normal value
    # z1 = math.sqrt(-float32(2.0) * math.log(u1)) * math.sin(TWO_PI_FLOAT32 * u2)
    return z0


@cuda.jit(float64(xoroshiro128p_type[:], int64), device=True)
def xoroshiro128p_normal_float64(states, index):
    '''Return a normally distributed float32 and advance states[index].

    The return value is drawn from a Gaussian of mean=0 and sigma=1 using the
    Box-Muller transform.  This advances the RNG sequence by two steps.

    Parameters
    ----------
    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: int64
    :param index: offset in states to update
    '''
    u1 = xoroshiro128p_uniform_float32(states, index)
    u2 = xoroshiro128p_uniform_float32(states, index)

    z0 = math.sqrt(-float64(2.0) * math.log(u1)) * math.cos(TWO_PI_FLOAT64 * u2)
    # discarding second normal value
    # z1 = math.sqrt(-float64(2.0) * math.log(u1)) * math.sin(TWO_PI_FLOAT64 * u2)
    return z0


@cuda.jit((xoroshiro128p_type[:], uint64, uint64))
def init_xoroshiro128p_states_kernel(states, seed, subsequence_start):
    # Only run this with a single thread and block
    n = states.shape[0]

    if n < 1:
        return  # assuming at least 1 state going forward

    init_xoroshiro128p_state(states, 0, seed)

    # advance to starting subsequence number
    for _ in range(subsequence_start):
        xoroshiro128p_jump(states, 0)

    # populate the rest of the array
    for i in range(1, n):
        states[i] = states[i - 1]  # take state of previous generator
        xoroshiro128p_jump(states, i) # and jump forward 2**64 steps


def init_xoroshiro128p_states(states, seed, subsequence_start=0, stream=0):
    '''Initialize RNG states on the GPU for parallel generators.
    
    This intializes the RNG states so that each state in the array corresponds
    subsequences in the separated by 2**64 steps from each other in the main
    sequence.  Therefore, as long no CUDA thread requests more than 2**64 random
    numbers, all of the RNG states produced by this function are guaranteed to
    be independent.

    The subsequence_start parameter can be used to advance the first RNG state
    by a multiple of 2**64 steps.

    Parameters
    ----------
    :type states: 1D DeviceNDArray, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type seed: uint64
    :param seed: starting seed for list of generators

    Returns
    -------
    float32 random number in range [0.0, 1.0)
    '''
    init_xoroshiro128p_states_kernel[1, 1, stream](states, seed, subsequence_start)


def create_xoroshiro128p_states(n, seed, subsequence_start=0, stream=0):
    '''Returns a new device array initialized for n random number generators.
    
    This intializes the RNG states so that each state in the array corresponds
    subsequences in the separated by 2**64 steps from each other in the main
    sequence.  Therefore, as long no CUDA thread requests more than 2**64 random
    numbers, all of the RNG states produced by this function are guaranteed to
    be independent.

    The subsequence_start parameter can be used to advance the first RNG state
    by a multiple of 2**64 steps.

    Parameters
    ----------
    :type n: int
    :param n: number of RNG states to create
    :type seed: uint64
    :param seed: starting seed for list of generators
    :type subsequence_start: uint64
    :param subsequence_start: 
    :type stream: CUDA stream 
    :param stream: stream to run initialization kernel on

    Returns
    -------
    float32 random number in range [0.0, 1.0)
    '''
    states = cuda.device_array(n, dtype=xoroshiro128p_dtype, stream=stream)
    init_xoroshiro128p_states(states, seed, subsequence_start, stream)
    return states


