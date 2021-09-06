"""Helper functions for distance matrix computations."""
import os
from math import sqrt
from numba import jit
from time import time

inline = os.environ.get("INLINE", "never")


class Timer(object):
    """
    Simple timer class.

    https://stackoverflow.com/a/5849861/13697228
    Usage
    -----
    with Timer("description"):
        # do stuff
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Enter the timer."""
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        """Exit the timer."""
        if self.name:
            print("[%s]" % self.name,)
        print(("Elapsed: {}\n").format(round((time() - self.tstart), 5)))


@jit(inline=inline)
def copy(a, out):
    """
    Copy a into out.

    Parameters
    ----------
    a : vector
        values to copy.
    out : vector
        values to copy into.

    Returns
    -------
    None.

    """
    for i in range(len(a)):
        i = int(i)
        out[i] = a[i]


@jit(inline=inline)
def insertionSort(arr):
    """
    Perform insertion sorting on a vector.

    Source: https://www.geeksforgeeks.org/insertion-sort/

    Parameters
    ----------
    arr : 1D array
        Array to be sorted in-place.

    Returns
    -------
    None.

    """
    for i in range(1, len(arr)):
        key = arr[i]

        # Move elements of arr[0..i-1], that are greater than key, to one
        # position ahead of their current position
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


@jit(inline=inline)
def insertionArgSort(arr, ids):
    """
    Perform insertion sorting on a vector and track the sorting indices.

    Source: https://www.geeksforgeeks.org/insertion-sort/

    Parameters
    ----------
    arr : 1D array
        Array to be sorted in-place.

    ids : 1D array
        Initialized array to fill with sorting indices.

    Returns
    -------
    None.

    """
    # fill ids
    for i in range(len(arr)):
        ids[i] = i

    for i in range(1, len(arr)):
        key = arr[i]
        id_key = ids[i]

        # Move elements of arr[0..i-1], that are greater than key, to one
        # position ahead of their current position
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            ids[j + 1] = ids[j]
            j -= 1
        arr[j + 1] = key
        ids[j + 1] = id_key


@jit(inline=inline)
def concatenate(vec, vec2, out):
    """
    Concatenate two vectors.

    It also reassigns the values of out back into vec and vec2 to prevent odd
    roundoff issues.

    Parameters
    ----------
    vec : vector
        First vector to concatenate.
    vec2 : vector
        Second vector to concatenate.
    out : vector
        Initialization of concatenated vector.

    Returns
    -------
    None.

    """
    # number of elements
    n = len(vec)
    n2 = len(vec2)
    # populate out
    for i in range(n):
        out[i] = vec[i]
    for i in range(n2):
        out[i + n] = vec2[i]


@jit(inline=inline)
def diff(vec, out):
    """
    Compute gradient.

    Parameters
    ----------
    vec : vector of floats
        The vector for which to calculate the gradient.
    out : vector of floats
        Initialization of output vector (one less element than vec) which
        contains the gradient.

    Returns
    -------
    None.

    """
    # number of elements
    n = len(vec) - 1
    # progressively add next elements
    for i in range(n):
        out[i] = vec[i + 1] - vec[i]

# maybe there's a faster implementation somewhere


@jit(inline=inline)
def bisect_right(a, v, ids):
    """
    Return indices where to insert items in v in list a, assuming a is sorted.

    Parameters
    ----------
    arr : 1D array
        Array for which to perform a binary search.
    v : 1D array
        Values to perform the binary search for.
    ids : 1D array
        Initialized list of indices, same size as v.

    Returns
    -------
    None.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there. Optional args lo
    (default 0) and hi (default len(a)) bound the slice of a to be searched.

    Source: modified from
    https://github.com/python/cpython/blob/43bab0537ceb6e2ca3597f8f3a3c79733b897434/Lib/bisect.py#L15-L35 # noqa
    """
    n = len(a)
    n2 = len(v)
    for i in range(n2):
        lo = 0
        mid = 0
        hi = n
        x = v[i]
        while lo < hi:
            mid = (lo + hi) // 2
            # Use __lt__ to match the logic in list.sort() and in heapq
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        ids[i] = lo


@jit(inline=inline)
def sort_by_indices(v, ids, out):
    """
    Sort v by ids and assign to out, as in out = v[ids].

    Parameters
    ----------
    v : vector
        values to sort.
    ids : vector
        indices to use for sorting.
    out : vector
        preallocated vector to put sorted values into.

    Returns
    -------
    None.

    """
    for i in range(len(ids)):
        idx = ids[i]
        out[i] = v[idx]


@jit(inline=inline)
def cumsum(vec, out):
    """
    Return the cumulative sum of the elements in a vector.

    Referenced https://stackoverflow.com/a/15889203/13697228

    Parameters
    ----------
    vec : 1D array
        Vector for which to compute the cumulative sum.

    out: 1D array
        Initialized vector which stores the cumulative sum.

    Returns
    -------
    out : 1D array
        The cumulative sum of elements in vec (same shape as vec).

    """
    # number of elements
    n = len(vec)
    # initialize
    total = 0.0
    # progressively add next elements
    for i in range(n):
        total += vec[i]
        out[i] = total


@jit(inline=inline)
def divide(v, b, out):
    """
    Divide a vector by a scalar.

    Parameters
    ----------
    v : vector
        vector numerator.
    b : float
        scalar denominator.
    out : vector
        initialized vector to populate with divided values.

    Returns
    -------
    None.

    """
    for i in range(len(v)):
        out[i] = v[i] / b


@jit(inline=inline)
def integrate(u_cdf, v_cdf, deltas, p):
    """
    Integrate between two vectors using a p-norm.

    Parameters
    ----------
    u_cdf : numeric vector
        First CDF.
    v_cdf : numeric vector
        Second CDF of same length as a.
    deltas : numeric vector
        The differences between the original vectors a and b.
    p : int, optional
        The finite power for which to compute the p-norm. The default is 2.

    If p == 1 or p == 2, use of power is avoided, which introduces an overhead
    # of about 15%
    Source: https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/stats/stats.py#L8404-L8488 # noqa
    However, without support for math.square or np.square (2021-08-19),
    math.pow is required for p == 2.

    Returns
    -------
    out : numeric scalar
        The p-norm of a and b.

    """
    n = len(u_cdf)
    out = 0.0
    if p == 1:
        for i in range(n):
            out += abs(u_cdf[i] - v_cdf[i]) * deltas[i]
    elif p == 2:
        for i in range(n):
            out += ((u_cdf[i] - v_cdf[i]) ** p) * deltas[i]
            out = sqrt(out)
    elif p > 2:
        for i in range(n):
            out += (u_cdf[i] - v_cdf[i]) ** p * deltas[i]
            out = out ** (1 / p)
    return out


# %% CODE GRAVEYARD
# from numba.core import config
# config_keys = dir(config)
# if "INLINE" in config_keys:
#     inline = config.INLINE
# else:
#     inline = "never"
