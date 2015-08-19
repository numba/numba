"""
Timsort implementation.  Mostly adapted from CPython's listobject.c.

For more information, see listsort.txt in CPython's source tree.
"""

from __future__ import print_function, absolute_import, division

import collections


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

MergeState = collections.namedtuple('MergeState',
                                    ('min_gallop', 'keys', 'values', 'pending'))

# A mergestate is a (min_gallop, keys, values, pending) tuple, where:
#  - *min_gallop* is an integer controlling when we get into galloping mode
#  - *keys* is a temp list for merging keys
#  - *values* is a temp list for merging values, if needed
#  - *pending* is a stack of (start, stop) tuples indicating pending runs to be merged

def merge_init(keys):
    """
    Initialize a MergeState for a non-keyed sort.
    """
    temp_keys = [keys[0]] * MERGESTATE_TEMP_SIZE
    temp_values = [False] * 0  # typed empty list
    pending = [(0, 0)] * 0     # typed empty list
    return MergeState(MIN_GALLOP, temp_keys, temp_values, pending)


def merge_getmem(ms, need):
    """
    Ensure enough temp memory for 'need' items is available.
    """
    alloced = len(ms.keys)
    if need <= alloced:
        return ms
    # Don't realloc!  That can cost cycles to copy the old data, but
    # we don't care what's in the block.
    temp_keys = [ms.keys[0]] * need
    if ms.values:
        temp_values = [ms.values[0]] * need
    else:
        temp_values = ms.values[0]
    return MergeState(ms.min_gallop, temp_keys, temp_values, ms.pending)


def merge_adjust_gallop(ms, new_gallop):
    """
    Modify the MergeState's min_gallop.
    """
    return MergeState(new_gallop, ms.keys, ms.values, ms.pending)


def sortslice_copy(dest_keys, dest_values, dest_start,
                   src_keys, src_values, src_start,
                   nitems):
    for i in range(nitems):
        dest_keys[dest_start + i] = src_keys[src_start + i]
    if src_values:
        for i in range(nitems):
            dest_values[dest_start + i] = src_values[src_start + i]


def merge_lo(ms, keys, values, ssa, na, ssb, nb):
    """
    Merge the na elements starting at ssa with the nb elements starting at
    ssb = ssa + na in a stable way, in-place.  na and nb must be > 0.
    Must also have that keys[ssa + na - 1] belongs at the end of the merge, and
    should have na <= nb.  See listsort.txt for more info.
    """
    assert na > 0 and nb > 0 and na <= nb, "merge_lo(): bad arguments"
    assert ssb == ssa + na, "merge_lo(): bad arguments"
    # First copy [ssa, ssa + na) into the temp space
    ms = merge_getmem(ms, na)
    sortslice_copy(ms.keys, ms.values, 0,
                   keys, values, ssa,
                   na)
    a_keys = ms.keys
    a_values = ms.values
    b_keys = keys
    b_values = values
    dest = ssa
    ssa = 0

    has_values = bool(a_values)
    min_gallop = ms.min_gallop

    # Now start merging into the space left from [ssa, ...)
    #keys[dest] = b_keys[ssb]
    #if has_values:
        #values[dest] = b_values[ssb]
    #dest += 1
    #ssb += 1
    #nb -= 1

    while nb > 0 and na > 1:
        # Do the straightforward thing until (if ever) one run
        # appears to win consistently.
        acount = 0
        bcount = 0

        while True:
            if LT(b_keys[ssb], a_keys[ssa]):
                keys[dest] = b_keys[ssb]
                if has_values:
                    values[dest] = b_values[ssb]
                dest += 1
                ssb += 1
                nb -= 1
                if nb == 0:
                    break
                # It's a B run
                bcount += 1
                acount = 0
                if bcount >= min_gallop:
                    break
            else:
                keys[dest] = a_keys[ssa]
                if has_values:
                    values[dest] = a_values[ssa]
                dest += 1
                ssa += 1
                na -= 1
                if na == 1:
                    break
                # It's a A run
                acount += 1
                bcount = 0
                if acount >= min_gallop:
                    break

        # TODO Gallop

    # Merge finished, now handle the remaining areas
    if nb == 0:
        # Only A remaining to copy
        sortslice_copy(keys, values, dest,
                       a_keys, a_values, ssa,
                       na)
    else:
        assert na == 1
        # The last element of A belongs at the end of the merge.
        sortslice_copy(keys, values, dest,
                       b_keys, b_values, ssb,
                       nb)
        keys[dest + nb] = a_keys[ssa]
        if has_values:
            values[dest + nb] = a_values[ssa]


#/* Merge the na elements starting at ssa with the nb elements starting at
 #* ssb.keys = ssa.keys + na in a stable way, in-place.  na and nb must be > 0.
 #* Must also have that ssa.keys[na-1] belongs at the end of the merge, and
 #* should have na <= nb.  See listsort.txt for more info.  Return 0 if
 #* successful, -1 if error.
 #*/
#static Py_ssize_t
#merge_lo(MergeState *ms, sortslice ssa, Py_ssize_t na,
         #sortslice ssb, Py_ssize_t nb)
#{
    #Py_ssize_t k;
    #sortslice dest;
    #int result = -1;            /* guilty until proved innocent */
    #Py_ssize_t min_gallop;

    #assert(ms && ssa.keys && ssb.keys && na > 0 && nb > 0);
    #assert(ssa.keys + na == ssb.keys);
    #if (MERGE_GETMEM(ms, na) < 0)
        #return -1;
    #sortslice_memcpy(&ms->a, 0, &ssa, 0, na);
    #dest = ssa;
    #ssa = ms->a;

    #sortslice_copy_incr(&dest, &ssb);
    #--nb;
    #if (nb == 0)
        #goto Succeed;
    #if (na == 1)
        #goto CopyB;

    #min_gallop = ms->min_gallop;
    #for (;;) {
        #Py_ssize_t acount = 0;          /* # of times A won in a row */
        #Py_ssize_t bcount = 0;          /* # of times B won in a row */

        #/* Do the straightforward thing until (if ever) one run
         #* appears to win consistently.
         #*/
        #for (;;) {
            #assert(na > 1 && nb > 0);
            #k = ISLT(ssb.keys[0], ssa.keys[0]);
            #if (k) {
                #if (k < 0)
                    #goto Fail;
                #sortslice_copy_incr(&dest, &ssb);
                #++bcount;
                #acount = 0;
                #--nb;
                #if (nb == 0)
                    #goto Succeed;
                #if (bcount >= min_gallop)
                    #break;
            #}
            #else {
                #sortslice_copy_incr(&dest, &ssa);
                #++acount;
                #bcount = 0;
                #--na;
                #if (na == 1)
                    #goto CopyB;
                #if (acount >= min_gallop)
                    #break;
            #}
        #}

        #/* One run is winning so consistently that galloping may
         #* be a huge win.  So try that, and continue galloping until
         #* (if ever) neither run appears to be winning consistently
         #* anymore.
         #*/
        #++min_gallop;
        #do {
            #assert(na > 1 && nb > 0);
            #min_gallop -= min_gallop > 1;
            #ms->min_gallop = min_gallop;
            #k = gallop_right(ssb.keys[0], ssa.keys, na, 0);
            #acount = k;
            #if (k) {
                #if (k < 0)
                    #goto Fail;
                #sortslice_memcpy(&dest, 0, &ssa, 0, k);
                #sortslice_advance(&dest, k);
                #sortslice_advance(&ssa, k);
                #na -= k;
                #if (na == 1)
                    #goto CopyB;
                #/* na==0 is impossible now if the comparison
                 #* function is consistent, but we can't assume
                 #* that it is.
                 #*/
                #if (na == 0)
                    #goto Succeed;
            #}
            #sortslice_copy_incr(&dest, &ssb);
            #--nb;
            #if (nb == 0)
                #goto Succeed;

            #k = gallop_left(ssa.keys[0], ssb.keys, nb, 0);
            #bcount = k;
            #if (k) {
                #if (k < 0)
                    #goto Fail;
                #sortslice_memmove(&dest, 0, &ssb, 0, k);
                #sortslice_advance(&dest, k);
                #sortslice_advance(&ssb, k);
                #nb -= k;
                #if (nb == 0)
                    #goto Succeed;
            #}
            #sortslice_copy_incr(&dest, &ssa);
            #--na;
            #if (na == 1)
                #goto CopyB;
        #} while (acount >= MIN_GALLOP || bcount >= MIN_GALLOP);
        #++min_gallop;           /* penalize it for leaving galloping mode */
        #ms->min_gallop = min_gallop;
    #}
#Succeed:
    #result = 0;
#Fail:
    #if (na)
        #sortslice_memcpy(&dest, 0, &ssa, 0, na);
    #return result;
#CopyB:
    #assert(na == 1 && nb > 0);
    #/* The last element of ssa belongs at the end of the merge. */
    #sortslice_memmove(&dest, 0, &ssb, 0, nb);
    #sortslice_copy(&dest, nb, &ssa, 0);
    #return 0;
#}
