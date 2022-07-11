import numpy as np

from numba.core.extending import register_jitable

from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
                                        UINT16_MAX, UINT8_MAX)
from numba.np.random.generator_core import next_uint32, next_uint64

# All following implementations are direct translations from:
# https://github.com/numpy/numpy/blob/7cfef93c77599bd387ecc6a15d186c5a46024dac/numpy/random/src/distributions/distributions.c


@register_jitable
def gen_mask(mask):
    mask = mask | (mask >> 1)
    mask = mask | (mask >> 2)
    mask = mask | (mask >> 4)
    mask = mask | (mask >> 8)
    mask = mask | (mask >> 16)
    mask = mask | (mask >> 32)
    return mask


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


@register_jitable
def buffered_bounded_lemire_uint8(bitgen, rng, bcnt, buf):
    # /*
    # * Uses Lemire's algorithm - https://arxiv.org/abs/1805.10941
    # *
    # * Note: `rng` should not be 0xFF. When this happens `rng_excl` becomes
    # * zero.
    # */
    rng_excl = (rng + 1) & 0xFF

    assert(rng != 0xFF)

    # /* Generate a scaled random number. */
    n, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)
    m = n * rng_excl

    # /* Rejection sampling to remove any bias */
    leftover = m & 0xFF

    if (leftover < rng_excl):
        # /* `rng_excl` is a simple upper bound for `threshold`. */
        threshold = ((UINT8_MAX - rng) % rng_excl) & 0xFF

        while (leftover < threshold):
            n, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)
            m = n * rng_excl
            leftover = m & 0xFF

    return (m >> 8) & 0xFF, bcnt, buf


@register_jitable
def buffered_bounded_lemire_uint16(bitgen, rng, bcnt, buf):
    # /*
    # * Uses Lemire's algorithm - https://arxiv.org/abs/1805.10941
    # *
    # * Note: `rng` should not be 0xFFFF. When this happens `rng_excl` becomes
    # * zero.
    # */
    rng_excl = (rng + 1) & 0xFFFF

    assert(rng != 0xFFFF)

    # /* Generate a scaled random number. */
    n, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
    m = n * rng_excl

    # /* Rejection sampling to remove any bias */
    leftover = m & 0xFFFF

    if (leftover < rng_excl):
        # /* `rng_excl` is a simple upper bound for `threshold`. */
        threshold = ((UINT16_MAX - rng) % rng_excl) & 0xFFFF

        while (leftover < threshold):
            n, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
            m = n * rng_excl
            leftover = m & 0xFFFF

    return (m >> 16) & 0xFFFF, bcnt, buf


@register_jitable
def buffered_bounded_lemire_uint32(bitgen, rng):
    rng_excl = rng + 1

    assert(rng != 0xFFFFFFFF)

    # Generate a scaled random number.
    m = next_uint32(bitgen) * rng_excl

    # Rejection sampling to remove any bias
    leftover = m & 0xFFFFFFFF

    if (leftover < rng_excl):
        # `rng_excl` is a simple upper bound for `threshold`.
        threshold = (UINT32_MAX - rng) % rng_excl

        while (leftover < threshold):
            m = next_uint32(bitgen) * rng_excl
            leftover = m & 0xFFFFFFFF

    return (m >> 32)


@register_jitable
def bounded_lemire_uint64(bitgen, rng):
    rng_excl = rng + 1

    assert(rng != 0xFFFFFFFFFFFFFFFF)

    m = next_uint64(bitgen) * rng_excl

    leftover = m & 0xFFFFFFFFFFFFFFFF

    if (leftover < rng_excl):
        threshold = (UINT64_MAX - rng) % rng_excl

        while (leftover < threshold):
            m = next_uint64(bitgen) * rng_excl
            leftover = m & 0xFFFFFFFFFFFFFFFF

    return (m >> 64)


#  Fills an array with cnt random npy_uint64 between off and off + rng
#  inclusive. The numbers wrap if rng is sufficiently large.
@register_jitable
def random_bounded_uint64_fill(bitgen, low, rng, mask, size, dtype):
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
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 0xFFFFFFFF:
        # /* Lemire32 doesn't support rng = 0xFFFFFFFF. */
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
    buf = 0
    bcnt = 0

    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 0xFFFF:
        # /* Lemire16 doesn't support rng = 0xFFFF. */
        for i in np.ndindex(size):
            val, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
            out[i] = low + val

    else:
        if mask is not None:
            # /* Smallest bit mask >= max */
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
    buf = 0
    bcnt = 0

    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 0xFF:
        # /* Lemire8 doesn't support rng = 0xFF. */
        for i in np.ndindex(size):
            val, bcnt, buf = buffered_uint8(bitgen, bcnt, buf)
            out[i] = low + val
    else:
        if mask is not None:
            # /* Smallest bit mask >= max */
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
    buf = 0
    bcnt = 0
    out = np.empty(size, dtype=dtype)
    for i in np.ndindex(size):
        val, bcnt, buf = buffered_bounded_bool(bitgen, low, rng,
                                               mask, bcnt, buf)
        out[i] = low + val
    return out


@register_jitable
def _randint_arg_check(low, high, endpoint, lower_bound, upper_bound):
    if high is None:
        high = low
        low = 0
    if not endpoint:
        high -= 1

    if low < lower_bound:
        raise ValueError("low is out of bounds")
    if high > upper_bound:
        raise ValueError("high is out of bounds")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low is greater than high in given interval")
    rng = (high - low)
    return low, rng
