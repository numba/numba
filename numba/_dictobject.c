/* The following is adapted from CPython3.7. */

/* Dictionary object implementation using a hash table */

/* The distribution includes a separate file, Objects/dictnotes.txt,
   describing explorations into dictionary design and optimization.
   It covers typical dictionary use patterns, the parameters for
   tuning dictionaries, and several ideas for possible optimizations.
*/

/* PyDictKeysObject

This implements the dictionary's hashtable.

As of Python 3.6, this is compact and ordered. Basic idea is described here:
* https://mail.python.org/pipermail/python-dev/2012-December/123028.html
* https://morepypy.blogspot.com/2015/01/faster-more-memory-efficient-and-more.html

layout:

+---------------+
| dk_refcnt     |
| dk_size       |
| dk_lookup     |
| dk_usable     |
| dk_nentries   |
+---------------+
| dk_indices    |
|               |
+---------------+
| dk_entries    |
|               |
+---------------+

dk_indices is actual hashtable.  It holds index in entries, or DKIX_EMPTY(-1)
or DKIX_DUMMY(-2).
Size of indices is dk_size.  Type of each index in indices is vary on dk_size:

* int8  for          dk_size <= 128
* int16 for 256   <= dk_size <= 2**15
* int32 for 2**16 <= dk_size <= 2**31
* int64 for 2**32 <= dk_size

dk_entries is array of PyDictKeyEntry.  It's size is USABLE_FRACTION(dk_size).
DK_ENTRIES(dk) can be used to get pointer to entries.

NOTE: Since negative value is used for DKIX_EMPTY and DKIX_DUMMY, type of
dk_indices entry is signed integer and int16 is used for table which
dk_size == 256.
*/


/*
The DictObject can be in one of two forms.

Either:
  A combined table:
    ma_values == NULL, dk_refcnt == 1.
    Values are stored in the me_value field of the PyDictKeysObject.
Or:
  A split table:
    ma_values != NULL, dk_refcnt >= 1
    Values are stored in the ma_values array.
    Only string (unicode) keys are allowed.
    All dicts sharing same key must have same insertion order.

There are four kinds of slots in the table (slot is index, and
DK_ENTRIES(keys)[index] if index >= 0):

1. Unused.  index == DKIX_EMPTY
   Does not hold an active (key, value) pair now and never did.  Unused can
   transition to Active upon key insertion.  This is each slot's initial state.

2. Active.  index >= 0, me_key != NULL and me_value != NULL
   Holds an active (key, value) pair.  Active can transition to Dummy or
   Pending upon key deletion (for combined and split tables respectively).
   This is the only case in which me_value != NULL.

3. Dummy.  index == DKIX_DUMMY  (combined only)
   Previously held an active (key, value) pair, but that was deleted and an
   active pair has not yet overwritten the slot.  Dummy can transition to
   Active upon key insertion.  Dummy slots cannot be made Unused again
   else the probe sequence in case of collision would have no way to know
   they were once active.

4. Pending. index >= 0, key != NULL, and value == NULL  (split only)
   Not yet inserted in split-table.
*/

/*
Preserving insertion order

It's simple for combined table.  Since dk_entries is mostly append only, we can
get insertion order by just iterating dk_entries.

One exception is .popitem().  It removes last item in dk_entries and decrement
dk_nentries to achieve amortized O(1).  Since there are DKIX_DUMMY remains in
dk_indices, we can't increment dk_usable even though dk_nentries is
decremented.

In split table, inserting into pending entry is allowed only for dk_entries[ix]
where ix == mp->ma_used. Inserting into other index and deleting item cause
converting the dict to the combined table.
*/

/* PyDict_MINSIZE is the starting size for any new dict.
 * 8 allows dicts with no more than 5 active entries; experiments suggested
 * this suffices for the majority of dicts (consisting mostly of usually-small
 * dicts created to pass keyword arguments).
 * Making this 8, rather than 4 reduces the number of resizes for most
 * dictionaries, without any significant extra memory use.
 */
#define DEBUG

#define NUMBA_INCREF(x)
#define NUMBA_DECREF(x)

#define PyDict_MINSIZE 8

#include "Python.h"
#include "_dictobject.h"
// #include "internal/pystate.h"
// #include "dict-common.h"
// #include "stringlib/eq.h"    /* to get unicode_eq() */

/*[clinic input]
class dict "PyDictObject *" "&PyDict_Type"
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=f157a5a0ce9589d6]*/


/*
To ensure the lookup algorithm terminates, there must be at least one Unused
slot (NULL key) in the table.
To avoid slowing down lookups on a near-full table, we resize the table when
it's USABLE_FRACTION (currently two-thirds) full.
*/

#define PERTURB_SHIFT 5

/*
Major subtleties ahead:  Most hash schemes depend on having a "good" hash
function, in the sense of simulating randomness.  Python doesn't:  its most
important hash functions (for ints) are very regular in common
cases:

  >>>[hash(i) for i in range(4)]
  [0, 1, 2, 3]

This isn't necessarily bad!  To the contrary, in a table of size 2**i, taking
the low-order i bits as the initial table index is extremely fast, and there
are no collisions at all for dicts indexed by a contiguous range of ints. So
this gives better-than-random behavior in common cases, and that's very
desirable.

OTOH, when collisions occur, the tendency to fill contiguous slices of the
hash table makes a good collision resolution strategy crucial.  Taking only
the last i bits of the hash code is also vulnerable:  for example, consider
the list [i << 16 for i in range(20000)] as a set of keys.  Since ints are
their own hash codes, and this fits in a dict of size 2**15, the last 15 bits
 of every hash code are all 0:  they *all* map to the same table index.

But catering to unusual cases should not slow the usual ones, so we just take
the last i bits anyway.  It's up to collision resolution to do the rest.  If
we *usually* find the key we're looking for on the first try (and, it turns
out, we usually do -- the table load factor is kept under 2/3, so the odds
are solidly in our favor), then it makes best sense to keep the initial index
computation dirt cheap.

The first half of collision resolution is to visit table indices via this
recurrence:

    j = ((5*j) + 1) mod 2**i

For any initial j in range(2**i), repeating that 2**i times generates each
int in range(2**i) exactly once (see any text on random-number generation for
proof).  By itself, this doesn't help much:  like linear probing (setting
j += 1, or j -= 1, on each loop trip), it scans the table entries in a fixed
order.  This would be bad, except that's not the only thing we do, and it's
actually *good* in the common cases where hash keys are consecutive.  In an
example that's really too small to make this entirely clear, for a table of
size 2**3 the order of indices is:

    0 -> 1 -> 6 -> 7 -> 4 -> 5 -> 2 -> 3 -> 0 [and here it's repeating]

If two things come in at index 5, the first place we look after is index 2,
not 6, so if another comes in at index 6 the collision at 5 didn't hurt it.
Linear probing is deadly in this case because there the fixed probe order
is the *same* as the order consecutive keys are likely to arrive.  But it's
extremely unlikely hash codes will follow a 5*j+1 recurrence by accident,
and certain that consecutive hash codes do not.

The other half of the strategy is to get the other bits of the hash code
into play.  This is done by initializing a (unsigned) vrbl "perturb" to the
full hash code, and changing the recurrence to:

    perturb >>= PERTURB_SHIFT;
    j = (5*j) + 1 + perturb;
    use j % 2**i as the next table index;

Now the probe sequence depends (eventually) on every bit in the hash code,
and the pseudo-scrambling property of recurring on 5*j+1 is more valuable,
because it quickly magnifies small differences in the bits that didn't affect
the initial index.  Note that because perturb is unsigned, if the recurrence
is executed often enough perturb eventually becomes and remains 0.  At that
point (very rarely reached) the recurrence is on (just) 5*j+1 again, and
that's certain to find an empty slot eventually (since it generates every int
in range(2**i), and we make sure there's always at least one empty slot).

Selecting a good value for PERTURB_SHIFT is a balancing act.  You want it
small so that the high bits of the hash code continue to affect the probe
sequence across iterations; but you want it large so that in really bad cases
the high-order hash bits have an effect on early iterations.  5 was "the
best" in minimizing total collisions across experiments Tim Peters ran (on
both normal and pathological cases), but 4 and 6 weren't significantly worse.

Historical: Reimer Behrends contributed the idea of using a polynomial-based
approach, using repeated multiplication by x in GF(2**n) where an irreducible
polynomial for each table size was chosen such that x was a primitive root.
Christian Tismer later extended that to use division by x instead, as an
efficient way to get the high bits of the hash code into play.  This scheme
also gave excellent collision statistics, but was more expensive:  two
if-tests were required inside the loop; computing "the next" index took about
the same number of operations but without as much potential parallelism
(e.g., computing 5*j can go on at the same time as computing 1+perturb in the
above, and then shifting perturb can be done while the table index is being
masked); and the PyDictObject struct required a member to hold the table's
polynomial.  In Tim's experiments the current scheme ran faster, produced
equally good collision statistics, needed less code & used less memory.

*/
//////////////////////////////////////////////////////////////////////////////


/* USABLE_FRACTION is the maximum dictionary load.
 * Increasing this ratio makes dictionaries more dense resulting in more
 * collisions.  Decreasing it improves sparseness at the expense of spreading
 * indices over more cache lines and at the cost of total memory consumed.
 *
 * USABLE_FRACTION must obey the following:
 *     (0 < USABLE_FRACTION(n) < n) for all n >= 2
 *
 * USABLE_FRACTION should be quick to calculate.
 * Fractions around 1/2 to 2/3 seem to work well in practice.
 */
#define USABLE_FRACTION(n) (((n) << 1)/3)

/* ESTIMATE_SIZE is reverse function of USABLE_FRACTION.
 * This can be used to reserve enough size to insert n entries without
 * resizing.
 */
#define ESTIMATE_SIZE(n)  (((n)*3+1) >> 1)

/* Alternative fraction that is otherwise close enough to 2n/3 to make
 * little difference. 8 * 2/3 == 8 * 5/8 == 5. 16 * 2/3 == 16 * 5/8 == 10.
 * 32 * 2/3 = 21, 32 * 5/8 = 20.
 * Its advantage is that it is faster to compute on machines with slow division.
 * #define USABLE_FRACTION(n) (((n) >> 1) + ((n) >> 2) - ((n) >> 3))
 */

/* GROWTH_RATE. Growth rate upon hitting maximum load.
 * Currently set to used*3.
 * This means that dicts double in size when growing without deletions,
 * but have more head room when the number of deletions is on a par with the
 * number of insertions.  See also bpo-17563 and bpo-33205.
 *
 * GROWTH_RATE was set to used*4 up to version 3.2.
 * GROWTH_RATE was set to used*2 in version 3.3.0
 * GROWTH_RATE was set to used*2 + capacity/2 in 3.4.0-3.6.0.
 */
#define GROWTH_RATE(d) ((d)->ma_used*3)

#define ENSURE_ALLOWS_DELETIONS(d) \
    if ((d)->ma_keys->dk_lookup == lookdict_unicode_nodummy) { \
        (d)->ma_keys->dk_lookup = lookdict_unicode; \
    }


#define DK_SIZE(dk) ((dk)->dk_size)
#if SIZEOF_VOID_P > 4
#define DK_IXSIZE(dk)                          \
    (DK_SIZE(dk) <= 0xff ?                     \
        1 : DK_SIZE(dk) <= 0xffff ?            \
            2 : DK_SIZE(dk) <= 0xffffffff ?    \
                4 : sizeof(int64_t))
#else
#define DK_IXSIZE(dk)                          \
    (DK_SIZE(dk) <= 0xff ?                     \
        1 : DK_SIZE(dk) <= 0xffff ?            \
            2 : sizeof(int32_t))
#endif
#define DK_ENTRIES(dk) \
    ((NumbaDictKeyEntry*)(&((int8_t*)((dk)->dk_indices))[DK_SIZE(dk) * DK_IXSIZE(dk)]))

#define DK_MASK(dk) (((dk)->dk_size)-1)

/*Global counter used to set ma_version_tag field of dictionary.
 * It is incremented each time that a dictionary is created and each
 * time that a dictionary is modified. */
static uint64_t pydict_global_version = 0;

#define DICT_NEXT_VERSION() (++pydict_global_version)


typedef enum {
    OK,
    ERR_NO_MEMORY,
} Status;



/* lookup indices.  returns DKIX_EMPTY, DKIX_DUMMY, or ix >=0 */
static inline Py_ssize_t
dk_get_index(NumbaDictKeysObject *keys, Py_ssize_t i)
{
    Py_ssize_t s = DK_SIZE(keys);
    Py_ssize_t ix;

    if (s <= 0xff) {
        int8_t *indices = (int8_t*)(keys->dk_indices);
        ix = indices[i];
    }
    else if (s <= 0xffff) {
        int16_t *indices = (int16_t*)(keys->dk_indices);
        ix = indices[i];
    }
#if SIZEOF_VOID_P > 4
    else if (s > 0xffffffff) {
        int64_t *indices = (int64_t*)(keys->dk_indices);
        ix = indices[i];
    }
#endif
    else {
        int32_t *indices = (int32_t*)(keys->dk_indices);
        ix = indices[i];
    }
    assert(ix >= DKIX_DUMMY);
    return ix;
}

int NumbaObject_Equal(NumbaObject *startkey, NumbaObject *key){
    if (startkey == key)
        return 1;
    else
        return 0;
}

/*
The basic lookup function used by all operations.
This is based on Algorithm D from Knuth Vol. 3, Sec. 6.4.
Open addressing is preferred over chaining since the link overhead for
chaining would be substantial (100% with typical malloc overhead).

The initial probe index is computed as hash mod the table size. Subsequent
probe indices are computed as explained earlier.

All arithmetic on hash should ignore overflow.

The details in this version are due to Tim Peters, building on many past
contributions by Reimer Behrends, Jyrki Alakuijala, Vladimir Marangozov and
Christian Tismer.

lookdict() is general-purpose, and may return DKIX_ERROR if (and only if) a
comparison raises an exception.
lookdict_unicode() below is specialized to string keys, comparison of which can
never raise an exception; that function can never return DKIX_ERROR when key
is string.  Otherwise, it falls back to lookdict().
lookdict_unicode_nodummy is further specialized for string keys that cannot be
the <dummy> value.
For both, when the key isn't found a DKIX_EMPTY is returned.
*/
static Py_ssize_t
lookdict(NumbaDictObject *mp, NumbaObject *key,
         Py_hash_t hash, NumbaObject **value_addr)
{
    size_t i, mask, perturb;
    NumbaDictKeysObject *dk;
    NumbaDictKeyEntry *ep0;

top:
    dk = mp->ma_keys;
    ep0 = DK_ENTRIES(dk);
    mask = DK_MASK(dk);
    perturb = hash;
    i = (size_t)hash & mask;

    for (;;) {
        Py_ssize_t ix = dk_get_index(dk, i);
        if (ix == DKIX_EMPTY) {
            *value_addr = NULL;
            return ix;
        }
        if (ix >= 0) {
            NumbaDictKeyEntry *ep = &ep0[ix];
            assert(ep->me_key != NULL);
            if (ep->me_key == key) {
                *value_addr = ep->me_value;
                return ix;
            }
            if (ep->me_hash == hash) {
                NumbaObject *startkey = ep->me_key;
                NUMBA_INCREF(startkey);
                int cmp = NumbaObject_Equal(startkey, key);
                NUMBA_DECREF(startkey);
                if (cmp < 0) {
                    *value_addr = NULL;
                    return DKIX_ERROR;
                }
                if (dk == mp->ma_keys && ep->me_key == startkey) {
                    if (cmp > 0) {
                        *value_addr = ep->me_value;
                        return ix;
                    }
                }
                else {
                    /* The dict was mutated, restart */
                    goto top;
                }
            }
        }
        perturb >>= PERTURB_SHIFT;
        i = (i*5 + perturb + 1) & mask;
    }
    exit(1); // unreachable
}



/* Search index of hash table from offset of entry table */
static Py_ssize_t
lookdict_index(NumbaDictKeysObject *k, Py_hash_t hash, Py_ssize_t index)
{
    size_t mask = DK_MASK(k);
    size_t perturb = (size_t)hash;
    size_t i = (size_t)hash & mask;

    for (;;) {
        Py_ssize_t ix = dk_get_index(k, i);
        if (ix == index) {
            return i;
        }
        if (ix == DKIX_EMPTY) {
            return DKIX_EMPTY;
        }
        perturb >>= PERTURB_SHIFT;
        i = mask & (i*5 + perturb + 1);
    }
    assert(0 && "unreachable");
}

/* write to indices. */
static inline void
dk_set_index(NumbaDictKeysObject *keys, Py_ssize_t i, Py_ssize_t ix)
{
    Py_ssize_t s = DK_SIZE(keys);

    assert(ix >= DKIX_DUMMY);

    if (s <= 0xff) {
        int8_t *indices = (int8_t*)(keys->dk_indices);
        assert(ix <= 0x7f);
        indices[i] = (char)ix;
    }
    else if (s <= 0xffff) {
        int16_t *indices = (int16_t*)(keys->dk_indices);
        assert(ix <= 0x7fff);
        indices[i] = (int16_t)ix;
    }
#if SIZEOF_VOID_P > 4
    else if (s > 0xffffffff) {
        int64_t *indices = (int64_t*)(keys->dk_indices);
        indices[i] = ix;
    }
#endif
    else {
        int32_t *indices = (int32_t*)(keys->dk_indices);
        assert(ix <= 0x7fffffff);
        indices[i] = (int32_t)ix;
    }
}


static
int Numba_new_keys_object(NumbaDictKeysObject **out, Py_ssize_t size)
{
    NumbaDictKeysObject *dk;
    Py_ssize_t es, usable;

    assert(size >= PyDict_MINSIZE);
    assert(IS_POWER_OF_2(size));

    usable = USABLE_FRACTION(size);
    if (size <= 0xff) {
        es = 1;
    }
    else if (size <= 0xffff) {
        es = 2;
    }
#if SIZEOF_VOID_P > 4
    else if (size <= 0xffffffff) {
        es = 4;
    }
#endif
    else {
        es = sizeof(Py_ssize_t);
    }


    dk = PyMem_RawMalloc(sizeof(NumbaDictKeysObject)
                            + es * size
                            + sizeof(NumbaDictKeyEntry) * usable);
    if (dk == NULL) {;
        return ERR_NO_MEMORY;
    }
    dk->dk_refcnt = 1;
    dk->dk_size = size;
    dk->dk_usable = usable;
    dk->dk_lookup = lookdict;
    dk->dk_nentries = 0;
    memset(&dk->dk_indices[0], 0xff, es * size);
    memset(DK_ENTRIES(dk), 0, sizeof(NumbaDictKeyEntry) * usable);

    // Set output
    *out = dk;
    return OK;
}



/*
Internal routine used by dictresize() to build a hashtable of entries.
*/
static void
build_indices(NumbaDictKeysObject *keys, NumbaDictKeyEntry *ep, Py_ssize_t n)
{
    size_t mask = (size_t)DK_SIZE(keys) - 1;
    for (Py_ssize_t ix = 0; ix != n; ix++, ep++) {
        Py_hash_t hash = ep->me_hash;
        size_t i = hash & mask;
        for (size_t perturb = hash; dk_get_index(keys, i) != DKIX_EMPTY;) {
            perturb >>= PERTURB_SHIFT;
            i = mask & (i*5 + perturb + 1);
        }
        dk_set_index(keys, i, ix);
    }
}

/*
Restructure the table by allocating a new table and reinserting all
items again.  When entries have been deleted, the new table may
actually be smaller than the old one.
If a table is split (its keys and hashes are shared, its values are not),
then the values are temporarily copied into the table, it is resized as
a combined table, then the me_value slots in the old table are NULLed out.
After resizing a table is always combined,
but can be resplit by make_keys_shared().
*/
static Status
dictresize(NumbaDictObject *mp, Py_ssize_t minsize)
{
    Py_ssize_t newsize, numentries;
    NumbaDictKeysObject *oldkeys;
    NumbaObject **oldvalues;
    NumbaDictKeyEntry *oldentries, *newentries;

    /* Find the smallest table size > minused. */
    for (newsize = PyDict_MINSIZE;
         newsize < minsize && newsize > 0;
         newsize <<= 1)
        ;
    if (newsize <= 0) {
        return ERR_NO_MEMORY;
    }

    oldkeys = mp->ma_keys;

    /* NOTE: Current odict checks mp->ma_keys to detect resize happen.
     * So we can't reuse oldkeys even if oldkeys->dk_size == newsize.
     * TODO: Try reusing oldkeys when reimplement odict.
     */

    /* Allocate a new table. */
    Status status = Numba_new_keys_object(&mp->ma_keys, newsize);
    if (status != OK) {
        mp->ma_keys = oldkeys;
        return status;
    }
    // New table must be large enough.
    assert(mp->ma_keys->dk_usable >= mp->ma_used);
    if (oldkeys->dk_lookup == lookdict)
        mp->ma_keys->dk_lookup = lookdict;

    numentries = mp->ma_used;
    oldentries = DK_ENTRIES(oldkeys);
    newentries = DK_ENTRIES(mp->ma_keys);
    oldvalues = mp->ma_values;
    if (oldvalues != NULL) {
        /* Convert split table into new combined table.
         * We must incref keys; we can transfer values.
         * Note that values of split table is always dense.
         */
        assert(0 && "do not do split table");
        for (Py_ssize_t i = 0; i < numentries; i++) {
            assert(oldvalues[i] != NULL);
            NumbaDictKeyEntry *ep = &oldentries[i];
            NumbaObject *key = ep->me_key;
            NUMBA_INCREF(key);
            newentries[i].me_key = key;
            newentries[i].me_hash = ep->me_hash;
            newentries[i].me_value = oldvalues[i];
        }

        NUMBA_DECREF(oldkeys);
        mp->ma_values = NULL;
        // if (oldvalues != empty_values) {
        PyMem_RawFree(oldvalues);
        // }
    }
    else {  // combined table.
        if (oldkeys->dk_nentries == numentries) {
            memcpy(newentries, oldentries, numentries * sizeof(NumbaDictKeyEntry));
        }
        else {
            NumbaDictKeyEntry *ep = oldentries;
            for (Py_ssize_t i = 0; i < numentries; i++) {
                while (ep->me_value == NULL)
                    ep++;
                newentries[i] = *ep++;
            }
        }

        assert(oldkeys->dk_lookup != lookdict_split);
        assert(oldkeys->dk_refcnt == 1);

        PyMem_RawFree(oldkeys);
    }

    build_indices(mp->ma_keys, newentries, numentries);
    mp->ma_keys->dk_usable -= numentries;
    mp->ma_keys->dk_nentries = numentries;
    return 0;
}



static int
insertion_resize(NumbaDictObject *mp)
{
    return dictresize(mp, GROWTH_RATE(mp));
}

/* Internal function to find slot for an item from its hash
   when it is known that the key is not present in the dict.

   The dict must be combined. */
static Py_ssize_t
find_empty_slot(PyDictKeysObject *keys, Py_hash_t hash)
{
    assert(keys != NULL);

    const size_t mask = DK_MASK(keys);
    size_t i = hash & mask;
    Py_ssize_t ix = dk_get_index(keys, i);
    for (size_t perturb = hash; ix >= 0;) {
        perturb >>= PERTURB_SHIFT;
        i = (i*5 + perturb + 1) & mask;
        ix = dk_get_index(keys, i);
    }
    return i;
}


/*
Internal routine to insert a new item into the table.
Used both by the internal resize routine and by the public insert routine.
Returns -1 if an error occurred, or 0 on success.
*/
static int
insertdict(NumbaDictObject *mp, NumbaObject *key, Py_hash_t hash, NumbaObject *value)
{
    printf("insertdict key=%p hash=%zx\n", key, hash);
    NumbaObject *old_value;
    NumbaDictKeyEntry *ep;

    NUMBA_INCREF(key);
    NUMBA_INCREF(value);

    assert(mp->ma_values);
    /* we don't do split tables
    //
    // if (mp->ma_values != NULL && !PyUnicode_CheckExact(key)) {
    //     if (insertion_resize(mp) < 0)
    //         goto Fail;
    // }
    */

    Py_ssize_t ix = mp->ma_keys->dk_lookup(mp, key, hash, &old_value);
    printf("insert index = %zd\n", ix);

    if (ix == DKIX_ERROR)
        goto Fail;

    // assert(PyUnicode_CheckExact(key) || mp->ma_keys->dk_lookup == lookdict);
    // MAINTAIN_TRACKING(mp, key, value);

    /* When insertion order is different from shared key, we can't share
     * the key anymore.  Convert this instance to combine table.
     */
    if (((ix >= 0 && old_value == NULL && mp->ma_used != ix) ||
         (ix == DKIX_EMPTY && mp->ma_used != mp->ma_keys->dk_nentries))) {
        if (insertion_resize(mp) < 0)
            goto Fail;
        ix = DKIX_EMPTY;
    }

    if (ix == DKIX_EMPTY) {
        /* Insert into new slot. */
        assert(old_value == NULL);
        if (mp->ma_keys->dk_usable <= 0) {
            /* Need to resize. */
            if (insertion_resize(mp) < 0)
                goto Fail;
        }
        Py_ssize_t hashpos = find_empty_slot(mp->ma_keys, hash);
        ep = &DK_ENTRIES(mp->ma_keys)[mp->ma_keys->dk_nentries];
        dk_set_index(mp->ma_keys, hashpos, mp->ma_keys->dk_nentries);
        ep->me_key = key;
        ep->me_hash = hash;
        if (mp->ma_values) {
            assert (mp->ma_values[mp->ma_keys->dk_nentries] == NULL);
            mp->ma_values[mp->ma_keys->dk_nentries] = value;
        }
        else {
            ep->me_value = value;
        }
        mp->ma_used++;
        mp->ma_version_tag = DICT_NEXT_VERSION();
        mp->ma_keys->dk_usable--;
        mp->ma_keys->dk_nentries++;
        assert(mp->ma_keys->dk_usable >= 0);
        // assert(_PyDict_CheckConsistency(mp));
        return 0;
    }

    // if (_PyDict_HasSplitTable(mp)) {
    //     mp->ma_values[ix] = value;
    //     if (old_value == NULL) {
    //         /* pending state */
    //         assert(ix == mp->ma_used);
    //         mp->ma_used++;
    //     }
    // }
    // else {
        assert(old_value != NULL);
        DK_ENTRIES(mp->ma_keys)[ix].me_value = value;
    // }

    mp->ma_version_tag = DICT_NEXT_VERSION();
    NUMBA_DECREF(old_value); /* which **CAN** re-enter (see issue #22653) */
    // assert(_PyDict_CheckConsistency(mp));
    NUMBA_DECREF(key);
    return OK;

Fail:
    NUMBA_DECREF(value);
    NUMBA_DECREF(key);
    return -1;
}


static Status
delitem_common(NumbaDictObject *mp, Py_hash_t hash, Py_ssize_t ix,
               PyObject *old_value)
{
    NumbaObject *old_key;
    NumbaDictKeyEntry *ep;

    Py_ssize_t hashpos = lookdict_index(mp->ma_keys, hash, ix);
    assert(hashpos >= 0);

    mp->ma_used--;
    mp->ma_version_tag = DICT_NEXT_VERSION();
    ep = &DK_ENTRIES(mp->ma_keys)[ix];
    dk_set_index(mp->ma_keys, hashpos, DKIX_DUMMY);
    // ENSURE_ALLOWS_DELETIONS(mp); // seems to only affect unicode
    old_key = ep->me_key;
    ep->me_key = NULL;
    ep->me_value = NULL;
    NUMBA_DECREF(old_key);
    NUMBA_DECREF(old_value);

    // assert(_PyDict_CheckConsistency(mp));
    return OK;
}


int
Numba_dict_new(NumbaDictObject **res) {
    NumbaDictObject* d = PyMem_RawMalloc(sizeof(NumbaDictObject));
    memset(d, 0, sizeof(NumbaDictObject));

    d->ma_used = 0;
    d->ma_version_tag = DICT_NEXT_VERSION();

    Numba_new_keys_object(&d->ma_keys, PyDict_MINSIZE);
    if (d->ma_keys == NULL) {
        return ERR_NO_MEMORY;
    }

    /* Set returned object */
    *res = d;
    return OK;
}



static
dict_keys_dump(NumbaDictObject *mp)
{
    PyObject *v;
    Py_ssize_t i, j;
    NumbaDictKeyEntry *ep;
    Py_ssize_t size, n, offset;
    PyObject **value_ptr;

    n = mp->ma_used;

    ep = DK_ENTRIES(mp->ma_keys);
    size = mp->ma_keys->dk_nentries;

    //// Never splitted
    // if (mp->ma_values) {
    //     value_ptr = mp->ma_values;
    //     offset = sizeof(PyObject *);
    // }
    // else {
        value_ptr = &ep[0].me_value;
        offset = sizeof(NumbaDictKeyEntry);
    // }
    printf("Key dump\n");
    for (i = 0, j = 0; i < size; i++) {
        if (*value_ptr != NULL) {
            NumbaObject *key = ep[i].me_key;
            Py_hash_t hash = ep[i].me_hash;
            printf("  key=%p hash=%zu value=%p\n", key, hash, *value_ptr);
            j++;
        }
        value_ptr = (PyObject **)(((char *)value_ptr) + offset);
    }
    assert(j == n);
}


static void show_status(Status status) {
    const char* msg = "<?>";
    switch (status) {
    case OK:
        msg = "OK";
        break;
    case ERR_NO_MEMORY:
        msg = "ERR_NO_MEMORY";
        break;
    }
    puts(msg);
}


NUMBA_EXPORT_FUNC(void)
test_dict() {
    puts("test_dict");
    NumbaDictObject* d;
    Status status = Numba_dict_new(&d);

    NumbaObject *key = 0xdead, *value = 0xbeef;
    Py_hash_t hash = 0xcafe;
    status = insertdict(d, key, hash, value);

    dict_keys_dump(d);

    key = 0xdeae; value = 0xbeed;
    status = insertdict(d, key, hash, value);

    dict_keys_dump(d);

    key = 0xdeaf; value = 0xbeee;
    status = insertdict(d, key, hash, value);
    dict_keys_dump(d);

    printf("d->ma_used = %d\n", (int)d->ma_used);

    printf("d->ma_keys = %p\n", d->ma_keys);
    printf("d->ma_values = %p\n", d->ma_values);

    NumbaObject *got_value = NULL;
    key = 0xdead;
    Py_ssize_t ix;
    ix = d->ma_keys->dk_lookup(d, key, hash, &got_value);
    printf("ix = %zd got_value=%p\n", ix, got_value);


    status = delitem_common(d, hash, ix, got_value);
    printf("ix = %zd got_value=%p\n", ix, got_value);

    dict_keys_dump(d);

    ix = d->ma_keys->dk_lookup(d, key, hash, &got_value);
    printf("ix = %zd got_value=%p\n", ix, got_value);


    key = 0xdeae;
    ix = d->ma_keys->dk_lookup(d, key, hash, &got_value);
    printf("ix = %zd got_value=%p\n", ix, got_value);


    dict_keys_dump(d);
    show_status(status);
}
