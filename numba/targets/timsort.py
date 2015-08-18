"""
Timsort implementation.  Mostly adapted from CPython's listobject.c.

For more information, see listsort.txt in CPython's source tree.
"""

from __future__ import print_function, absolute_import, division


def LT(a, b):
    """
    Trivial comparison function between two keys.  This is factored out to
    make it clear where comparison occurs.
    """
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
        # NOTE: bisection only wins over linear search if the comparison
        # function is much more expensive than simply moving data.
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


def gallop_left(key, a, start, stop, hint):
    """
    Locate the proper position of key in a sorted vector; if the vector contains
    an element equal to key, return the position immediately to the left of
    the leftmost equal element.  [gallop_right() does the same except returns
    the position to the right of the rightmost equal element (if any).]

    "a" is a sorted vector with stop elements, starting at a[start].
    stop must be > start.

    "hint" is an index at which to begin the search, start <= hint < stop.
    The closer hint is to the final result, the faster this runs.

    The return value is the int k in start..stop such that

        a[k-1] < key <= a[k]

    pretending that a[start-1] is minus infinity and a[stop] is plus infinity.
    IOW, key belongs at index k; or, IOW, the first k elements of a should
    precede key, and the last stop-start-k should follow key.

    See listsort.txt for info on the method.
    """
    assert stop > start, "gallop_left(): stop <= start"
    assert hint >= start and hint < stop, "gallop_left(): hint not in [start, stop)"
    n = stop - start

    # First, gallop from the hint to find a "good" subinterval for bisecting
    lastofs = 0
    ofs = 1
    if LT(a[hint], key):
        # a[hint] < key => gallop right, until
        #                  a[hint + lastofs] < key <= a[hint + ofs]
        maxofs = stop - hint
        while ofs < maxofs:
            if LT(a[hint + ofs], key):
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    # Int overflow
                    ofs = maxofs
            else:
                # key <= a[hint + ofs]
                break
        if ofs > maxofs:
            ofs = maxofs
        # Translate back to offsets relative to a[0]
        lastofs += hint
        ofs += hint
    else:
        # key <= a[hint] => gallop left, until
        #                   a[hint - ofs] < key <= a[hint - lastofs]
        maxofs = hint - start + 1
        while ofs < maxofs:
            if LT(a[hint - ofs], key):
                break
            else:
                # key <= a[hint - ofs]
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    # Int overflow
                    ofs = maxofs
        if ofs > maxofs:
            ofs = maxofs
        # Translate back to positive offsets relative to a[0]
        lastofs, ofs = hint - ofs, hint - lastofs

    assert start - 1 <= lastofs and lastofs < ofs and ofs <= stop
    # Now a[lastofs] < key <= a[ofs], so key belongs somewhere to the
    # right of lastofs but no farther right than ofs.  Do a binary
    # search, with invariant a[lastofs-1] < key <= a[ofs].
    lastofs += 1
    while lastofs < ofs:
        m = lastofs + ((ofs - lastofs) >> 1)
        if LT(a[m], key):
            # a[m] < key
            lastofs = m + 1
        else:
            # key <= a[m]
            ofs = m
    # Now lastofs == ofs, so a[ofs - 1] < key <= a[ofs]
    return ofs


def gallop_right(key, a, start, stop, hint):
    """
    Exactly like gallop_left(), except that if key already exists in a[start:stop],
    finds the position immediately to the right of the rightmost equal value.

    The return value is the int k in start..stop such that

        a[k-1] <= key < a[k]

    The code duplication is massive, but this is enough different given that
    we're sticking to "<" comparisons that it's much harder to follow if
    written as one routine with yet another "left or right?" flag.
    """
    assert stop > start, "gallop_right(): stop <= start"
    assert hint >= start and hint < stop, "gallop_right(): hint not in [start, stop)"
    n = stop - start

    # First, gallop from the hint to find a "good" subinterval for bisecting
    lastofs = 0
    ofs = 1
    if LT(key, a[hint]):
        # key < a[hint] => gallop left, until
        #                  a[hint - ofs] <= key < a[hint - lastofs]
        maxofs = hint - start + 1
        while ofs < maxofs:
            if LT(key, a[hint - ofs]):
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    # Int overflow
                    ofs = maxofs
            else:
                # a[hint - ofs] <= key
                break
        if ofs > maxofs:
            ofs = maxofs
        # Translate back to positive offsets relative to a[0]
        lastofs, ofs = hint - ofs, hint - lastofs
    else:
        # a[hint] <= key -- gallop right, until
        # a[hint + lastofs] <= key < a[hint + ofs]
        maxofs = stop - hint
        while ofs < maxofs:
            if LT(key, a[hint + ofs]):
                break
            else:
                # a[hint + ofs] <= key
                lastofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    # Int overflow
                    ofs = maxofs
        if ofs > maxofs:
            ofs = maxofs
        # Translate back to offsets relative to a[0]
        lastofs += hint
        ofs += hint

    assert start - 1 <= lastofs and lastofs < ofs and ofs <= stop
    # Now a[lastofs] <= key < a[ofs], so key belongs somewhere to the
    # right of lastofs but no farther right than ofs.  Do a binary
    # search, with invariant a[lastofs-1] <= key < a[ofs].
    lastofs += 1
    while lastofs < ofs:
        m = lastofs + ((ofs - lastofs) >> 1)
        if LT(key, a[m]):
            # key < a[m]
            ofs = m
        else:
            # a[m] <= key
            lastofs = m + 1
    # Now lastofs == ofs, so a[ofs - 1] <= key < a[ofs]
    return ofs


def merge_compute_minrun(n):
    """
    Compute a good value for the minimum run length; natural runs shorter
    than this are boosted artificially via binary insertion.

    If n < 64, return n (it's too small to bother with fancy stuff).
    Else if n is an exact power of 2, return 32.
    Else return an int k, 32 <= k <= 64, such that n/k is close to, but
    strictly less than, an exact power of 2.

    See listsort.txt for more info.
    """
    r = 0
    assert n >= 0
    while n >= 64:
        r |= n & 1
        n >>= 1
    return n + r


MIN_GALLOP = 7
MERGESTATE_TEMP_SIZE = 256

# A mergestate is a (min_gallop, temp_keys, temp_values, pending) tuple, where:
# - min_gallop is an integer controlling when we get into galloping mode
# - temp_keys is a temp list for merging keys
# - temp_values is a temp list for merging values, if needed
# - pending is a stack of (start, stop) tuples indicating pending runs to be merged

def merge_init(keys, values):
    temp_keys = [keys[0]] * MERGESTATE_TEMP_SIZE
    if values:
        temp_values = [values[0]] * MERGESTATE_TEMP_SIZE
    else:
        temp_values = values[:]  # for typing
    pending = [(0, 0)] * 0
    return (MIN_GALLOP, temp_keys, temp_values, pending)

