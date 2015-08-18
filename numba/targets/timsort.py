
from __future__ import print_function, absolute_import, division


def LT(a, b):
    return a < b


def binarysort(keys, values, lo, hi, start):
    """
    binarysort is the best method for sorting small arrays: it does
    few compares, but can do data movement quadratic in the number of
    elements.
    [lo, hi) is a contiguous slice of a list, and is sorted via
    binary insertion.  This sort is stable.
    On entry, must have lo <= start <= hi, and that [lo, start) is already
    sorted (pass start == lo if you don't know!).
    """
    assert lo <= start and start <= hi, "Bad input for binarysort()"
    if lo == start:
        start += 1
    while start < hi:
        pivot = keys[start]
        # Bisect to find where to insert `pivot`
        l = lo
        r = start
        # Invariants:
        # pivot >= all in [lo, l).
        # pivot  < all in [r, start).
        # The second is vacuously true at the start.
        while l < r:
            p = l + ((r - l) >> 1)
            if LT(pivot, keys[p]):
                r = p
            else:
                l = p+1

        # The invariants still hold, so pivot >= all in [lo, l) and
        # pivot < all in [l, start), so pivot belongs at l.  Note
        # that if there are elements equal to pivot, l points to the
        # first slot after them -- that's why this sort is stable.
        # Slide over to make room (aka memmove()).
        for p in range(start, l, -1):
            keys[p] = keys[p - 1]
        keys[l] = pivot
        if values:
            pivot_val = values[start]
            for p in range(start, l, -1):
                values[p] = values[p - 1]
            values[l] = pivot_val

        start += 1


def count_run(keys, lo, hi):
    """
    Return the length of the run beginning at lo, in the slice [lo, hi).
    lo < hi is required on entry.  "A run" is the longest ascending sequence, with

        lo[0] <= lo[1] <= lo[2] <= ...

    or the longest descending sequence, with

        lo[0] > lo[1] > lo[2] > ...

    A tuple (length, descending) is returned, where boolean *descending*
    is set to 0 in the former case, or to 1 in the latter.
    For its intended use in a stable mergesort, the strictness of the defn of
    "descending" is needed so that the caller can safely reverse a descending
    sequence without violating stability (strict > ensures there are no equal
    elements to get out of order).
    """
    assert lo < hi, "Bad input for count_run()"
    if lo + 1 == hi:
        # Trivial 1-long run
        return 1, False
    if LT(keys[lo + 1], keys[lo]):
        # Descending run
        for k in range(lo + 2, hi):
            if not LT(keys[k], keys[k - 1]):
                return k - lo, True
        return hi - lo, True
    else:
        # Ascending run
        for k in range(lo + 2, hi):
            if LT(keys[k], keys[k - 1]):
                return k - lo, False
        return hi - lo, False

