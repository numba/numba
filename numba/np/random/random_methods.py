import numpy as np

from numba import uint64, uint32, uint16
from numba.core.extending import register_jitable

from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
                                        UINT16_MAX, UINT8_MAX)
from numba.np.random.generator_core import next_uint32, next_uint64

# All following implementations are direct translations from:
# https://github.com/numpy/numpy/blob/7cfef93c77599bd387ecc6a15d186c5a46024dac/numpy/random/src/distributions/distributions.c


@register_jitable
def buffered_bounded_bool(bitgen, off, rng, mask, bcnt, buf):
    if (rng == 0):
        return off, bcnt, buf
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 31
    else:
        buf >>= 1
        bcnt -= 1

    return ((buf & 1) != 0), bcnt, buf


@register_jitable
def buffered_uint8(bitgen, bcnt, buf):
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 3
    else:
        buf >>= 8
        bcnt -= 1

    return buf & 0xff, bcnt, buf


@register_jitable
def buffered_uint16(bitgen, bcnt, buf):
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 1
    else:
        buf >>= 16
        bcnt -= 1

    return buf & 0xffff, bcnt, buf


@register_jitable
def buffered_bounded_masked_uint32(bitgen, rng, mask):
    val = (next_uint32(bitgen) & mask)
    while (val > rng):
        val = (next_uint32(bitgen) & mask)
    return val


@register_jitable
def buffered_bounded_masked_uint16(bitgen, rng, mask, bcnt, buf):
    val, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)

    while (val & mask) > rng:
        val, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)

    return val, bcnt, buf


@register_jitable
def buffered_bounded_masked_uint8(bitgen, rng, mask, bcnt, buf):
    val, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)

    while (val & mask) > rng:
        val, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)

    return val, bcnt, buf


@register_jitable
def bounded_masked_uint64(bitgen, rng, mask):
    val = (next_uint64(bitgen) & mask)
    while (val > rng):
        val = (next_uint64(bitgen) & mask)
    return val


# The following implementations use Lemire's algorithm:
# https://arxiv.org/abs/1805.10941
@register_jitable
def buffered_bounded_lemire_uint8(bitgen, rng, bcnt, buf):
    """
    Generates a random unsigned 8 bit integer bounded
    within a given interval using Lemire's rejection.

    The buffer acts as a storage for a 32 bit integer
    drawn from the associated BitGenerator so that we
    can generate multiple integers of smaller bitsize
    from a single draw of the BitGenerator.
    """
    # Note: `rng` should not be 0xFF. When this happens `rng_excl` becomes
    # zero.
    rng_excl = (rng + 1) & 0xFF

    assert(rng != 0xFF)

    # Generate a scaled random number.
    n, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)
    m = uint16(n * rng_excl)

    # Rejection sampling to remove any bias
    leftover = m & 0xFF

    if (leftover < rng_excl):
        # `rng_excl` is a simple upper bound for `threshold`.
        threshold = ((UINT8_MAX - rng) % rng_excl) & 0xFF

        while (leftover < threshold):
            n, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)
            m = uint16(n * rng_excl)
            leftover = m & 0xFF

    return m >> 8, bcnt, buf


@register_jitable
def buffered_bounded_lemire_uint16(bitgen, rng, bcnt, buf):
    """
    Generates a random unsigned 16 bit integer bounded
    within a given interval using Lemire's rejection.

    The buffer acts as a storage for a 32 bit integer
    drawn from the associated BitGenerator so that we
    can generate multiple integers of smaller bitsize
    from a single draw of the BitGenerator.
    """
    # Note: `rng` should not be 0xFFFF. When this happens `rng_excl` becomes
    # zero.
    rng_excl = (rng + 1) & 0xFFFF

    assert(rng != 0xFFFF)

    # Generate a scaled random number.
    n, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
    m = uint32(n * rng_excl)

    # Rejection sampling to remove any bias
    leftover = m & 0xFFFF

    if (leftover < rng_excl):
        # `rng_excl` is a simple upper bound for `threshold`.
        threshold = ((UINT16_MAX - rng) % rng_excl) & 0xFFFF

        while (leftover < threshold):
            n, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
            m = uint32(n * rng_excl)
            leftover = m & 0xFFFF

    return m >> 16, bcnt, buf


@register_jitable
def buffered_bounded_lemire_uint32(bitgen, rng):
    """
    Generates a random unsigned 32 bit integer bounded
    within a given interval using Lemire's rejection.
    """
    rng_excl = rng + 1

    assert(rng != 0xFFFFFFFF)

    # Generate a scaled random number.
    m = uint64(next_uint32(bitgen) * rng_excl)

    # Rejection sampling to remove any bias
    leftover = m & 0xFFFFFFFF

    if (leftover < rng_excl):
        # `rng_excl` is a simple upper bound for `threshold`.
        threshold = (UINT32_MAX - rng) % rng_excl

        while (leftover < threshold):
            m = uint64(next_uint32(bitgen) * rng_excl)
            leftover = m & 0xFFFFFFFF

    return (m >> 32)


@register_jitable
def bounded_lemire_uint64(bitgen, rng):
    """
    Generates a random unsigned 64 bit integer bounded
    within a given interval using Lemire's rejection.
    """
    rng_excl = rng + uint64(1)

    assert(rng != 0xFFFFFFFFFFFFFFFF)

    x = next_uint64(bitgen)

    leftover = uint64(x) * uint64(rng_excl)

    if (leftover < rng_excl):
        threshold = (UINT64_MAX - rng) % rng_excl

        while (leftover < threshold):
            x = next_uint64(bitgen)
            leftover = uint64(x) * uint64(rng_excl)

    x0 = x & uint64(0xFFFFFFFF)
    x1 = x >> 32
    rng_excl0 = rng_excl & uint64(0xFFFFFFFF)
    rng_excl1 = rng_excl >> 32
    w0 = x0 * rng_excl0
    t = x1 * rng_excl0 + (w0 >> 32)
    w1 = t & uint64(0xFFFFFFFF)
    w2 = t >> 32
    w1 += x0 * rng_excl1
    m1 = x1 * rng_excl1 + w2 + (w1 >> 32)

    return m1


# Fills an array with cnt random npy_uint64 between off and off + rng
# inclusive. The numbers wrap if rng is sufficiently large.
@register_jitable
def random_bounded_uint64_fill(bitgen, low, rng, mask, size, dtype):
    """
    Fills an array of given size with 64 bit integers
    bounded by given interval.
    """
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng <= 0xFFFFFFFF:
        if (rng == 0xFFFFFFFF):
            for i in np.ndindex(size):
                out[i] = low + next_uint32(bitgen)
        else:
            if mask is not None:
                for i in np.ndindex(size):
                    out[i] = low + \
                        buffered_bounded_masked_uint32(bitgen,
                                                       rng, mask)
            else:
                for i in np.ndindex(size):
                    out[i] = low + buffered_bounded_lemire_uint32(bitgen, rng)

    elif (rng == 0xFFFFFFFFFFFFFFFF):
        for i in np.ndindex(size):
            out[i] = low + next_uint64(bitgen)
    else:
        if mask is not None:
            for i in np.ndindex(size):
                out[i] = low + bounded_masked_uint64(bitgen, rng, mask)
        else:
            for i in np.ndindex(size):
                out[i] = low + bounded_lemire_uint64(bitgen, rng)

    return out


@register_jitable
def random_bounded_uint32_fill(bitgen, low, rng, mask, size, dtype):
    """
    Fills an array of given size with 32 bit integers
    bounded by given interval.
    """
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 0xFFFFFFFF:
        # Lemire32 doesn't support rng = 0xFFFFFFFF.
        for i in np.ndindex(size):
            out[i] = low + next_uint32(bitgen)
    else:
        if mask is not None:
            for i in np.ndindex(size):
                out[i] = low + buffered_bounded_masked_uint32(bitgen, rng, mask)
        else:
            for i in np.ndindex(size):
                out[i] = low + buffered_bounded_lemire_uint32(bitgen, rng)
    return out


@register_jitable
def random_bounded_uint16_fill(bitgen, low, rng, mask, size, dtype):
    """
    Fills an array of given size with 16 bit integers
    bounded by given interval.
    """
    buf = 0
    bcnt = 0

    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 0xFFFF:
        # Lemire16 doesn't support rng = 0xFFFF.
        for i in np.ndindex(size):
            val, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
            out[i] = low + val

    else:
        if mask is not None:
            # Smallest bit mask >= max
            for i in np.ndindex(size):
                val, bcnt, buf = \
                    buffered_bounded_masked_uint16(bitgen, rng,
                                                   mask, bcnt, buf)
                out[i] = low + val
        else:
            for i in np.ndindex(size):
                val, bcnt, buf = \
                    buffered_bounded_lemire_uint16(bitgen, rng,
                                                   bcnt, buf)
                out[i] = low + val
    return out


@register_jitable
def random_bounded_uint8_fill(bitgen, low, rng, mask, size, dtype):
    """
    Fills an array of given size with 8 bit integers
    bounded by given interval.
    """
    buf = 0
    bcnt = 0

    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 0xFF:
        # Lemire8 doesn't support rng = 0xFF.
        for i in np.ndindex(size):
            val, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)
            out[i] = low + val
    else:
        if mask is not None:
            # Smallest bit mask >= max
            for i in np.ndindex(size):
                val, bcnt, buf = \
                    buffered_bounded_masked_uint8(bitgen, rng,
                                                  mask, bcnt, buf)
                out[i] = low + val
        else:
            for i in np.ndindex(size):
                val, bcnt, buf = \
                    buffered_bounded_lemire_uint8(bitgen, rng,
                                                  bcnt, buf)
                out[i] = low + val
    return out


@register_jitable
def random_bounded_bool_fill(bitgen, low, rng, mask, size, dtype):
    """
    Fills an array of given size with boolean values.
    """
    buf = 0
    bcnt = 0
    out = np.empty(size, dtype=dtype)
    for i in np.ndindex(size):
        val, bcnt, buf = buffered_bounded_bool(bitgen, low, rng,
                                               mask, bcnt, buf)
        out[i] = low + val
    return out


@register_jitable
def _randint_arg_check(low, high, lower_bound, upper_bound):
    """
    Checks if low and high are correctly within the bounds
    for the given datatype.
    """

    if low < lower_bound:
        raise ValueError("low is out of bounds")
    if high > upper_bound:
        raise ValueError("high is out of bounds")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low is greater than high in given interval")
