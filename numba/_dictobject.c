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

  (Numba dev notes: split table logic is removed)

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


/* D_MINSIZE (adapted from PyDict_MINSIZE)
 * is the starting size for any new dict.
 * 8 allows dicts with no more than 5 active entries; experiments suggested
 * this suffices for the majority of dicts (consisting mostly of usually-small
 * dicts created to pass keyword arguments).
 * Making this 8, rather than 4 reduces the number of resizes for most
 * dictionaries, without any significant extra memory use.
 */
#define D_MINSIZE 8

#include "_dictobject.h"

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

#define DKIX_EMPTY (-1)
#define DKIX_DUMMY (-2)  /* Used internally */
#define DKIX_ERROR (-3)

typedef enum {
    OK = 0,
    ERR_NO_MEMORY = -1,
    ERR_DICT_MUTATED = -2,
    ERR_ITER_EXHAUSTED = -3,
} Status;


#ifndef NDEBUG
static
int mem_cmp_zeros(void *obj, size_t n){
    int diff = 0;
    char *mem = obj;
    for (char *it = mem; it < mem + n; ++it) {
        if (*it != 0) diff += 1;
    }
    return diff;
}
#endif

#define D_MASK(dk) ((dk)->size-1)
#define D_GROWTH_RATE(d) ((d)->used*3)

static int
ix_size(Py_ssize_t size) {
    if ( size < 0xff ) return 1;
    if ( size < 0xffff ) return 2;
    if ( size < 0xffffffff ) return 4;
    return sizeof(int64_t);
}

/* Align size *sz* to pointer width */
static Py_ssize_t
align(Py_ssize_t sz) {
    Py_ssize_t alignment = sizeof(void*);
    return sz + (alignment - sz % alignment) % alignment;
}

/* Align pointer *ptr* to pointer size */
static void*
palign(void *ptr) {
    return (void*)align((size_t)ptr);
}

/* lookup indices.  returns DKIX_EMPTY, DKIX_DUMMY, or ix >=0 */
static inline Py_ssize_t
get_index(NB_DictKeys *dk, Py_ssize_t i)
{
    Py_ssize_t s = dk->size;
    Py_ssize_t ix;

    if (s <= 0xff) {
        int8_t *indices = (int8_t*)(dk->indices);
        assert (i < dk->size);
        ix = indices[i];
    }
    else if (s <= 0xffff) {
        int16_t *indices = (int16_t*)(dk->indices);
        ix = indices[i];
    }
#if SIZEOF_VOID_P > 4
    else if (s > 0xffffffff) {
        int64_t *indices = (int64_t*)(dk->indices);
        ix = indices[i];
    }
#endif
    else {
        int32_t *indices = (int32_t*)(dk->indices);
        ix = indices[i];
    }
    assert(ix >= DKIX_DUMMY);
    return ix;
}

/* write to indices. */
static inline void
set_index(NB_DictKeys *dk, Py_ssize_t i, Py_ssize_t ix)
{
    Py_ssize_t s = dk->size;

    assert(ix >= DKIX_DUMMY);

    if (s <= 0xff) {
        int8_t *indices = (int8_t*)(dk->indices);
        assert(ix <= 0x7f);
        indices[i] = (char)ix;
    }
    else if (s <= 0xffff) {
        int16_t *indices = (int16_t*)(dk->indices);
        assert(ix <= 0x7fff);
        indices[i] = (int16_t)ix;
    }
#if SIZEOF_VOID_P > 4
    else if (s > 0xffffffff) {
        int64_t *indices = (int64_t*)(dk->indices);
        indices[i] = ix;
    }
#endif
    else {
        int32_t *indices = (int32_t*)(dk->indices);
        assert(ix <= 0x7fffffff);
        indices[i] = (int32_t)ix;
    }
}


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

void
numba_dictkeys_free(NB_DictKeys *dk) {
    free(dk);
}

void
numba_dict_free(NB_Dict *d) {
    numba_dictkeys_free(d->keys);
    free(d);
}

Py_ssize_t
numba_dict_length(NB_Dict *d) {
    return d->used;
}


static NB_DictEntry*
get_entry(NB_DictKeys *dk, Py_ssize_t idx) {
    assert (idx < dk->size);
    Py_ssize_t offset = idx * dk->entry_size;
    char *ptr = dk->indices + dk->entry_offset + offset;
    return (NB_DictEntry*)ptr;
}

static void
zero_key(NB_DictKeys *dk, char *data){
    memset(data, 0, dk->key_size);
}

static void
zero_val(NB_DictKeys *dk, char *data){
    memset(data, 0, dk->val_size);
}

static void
copy_key(NB_DictKeys *dk, char *dst, const char *src){
    memcpy(dst, src, dk->key_size);
}

static void
copy_val(NB_DictKeys *dk, char *dst, const char *src){
    memcpy(dst, src, dk->val_size);
}

/* Returns -1 for error; 0 for not equal; 1 for equal */
static int
key_equal(NB_DictKeys *dk, const char *lhs, const char *rhs) {
    /* XXX: should use user provided Equality */
    return memcmp(lhs, rhs, dk->key_size) == 0;
}

static char *
entry_get_key(NB_DictKeys *dk, NB_DictEntry* entry) {
    char * out = entry->keyvalue;
    assert (out == palign(out));
    return out;
}

static char *
entry_get_val(NB_DictKeys *dk, NB_DictEntry* entry) {
    char * out = entry_get_key(dk, entry) + align(dk->key_size);
    assert (out == palign(out));
    return out;
}

/* Allocate new dictionary keys */
int
numba_dictkeys_new(NB_DictKeys **out, Py_ssize_t size, Py_ssize_t key_size, Py_ssize_t val_size) {
    Py_ssize_t usable = USABLE_FRACTION(size);
    Py_ssize_t index_size = ix_size(size);
    Py_ssize_t entry_size = align(sizeof(NB_DictEntry) + align(key_size) + align(val_size));
    Py_ssize_t entry_offset = align(index_size * size);
    Py_ssize_t alloc_size = sizeof(NB_DictKeys) + entry_offset + entry_size * usable;

    NB_DictKeys *dk = palign(malloc(align(alloc_size)));
    if (!dk) return ERR_NO_MEMORY;

    assert ( size >= D_MINSIZE );

    dk->size = size;
    dk->usable = usable;
    dk->nentries = 0;
    dk->key_size = key_size;
    dk->val_size = val_size;
    dk->entry_offset = entry_offset;
    dk->entry_size = entry_size;

    assert (palign(dk->indices) == dk->indices );
    memset(dk->indices, 0xff, entry_offset);
    /* Ensure hash is (-1) for empty */
    memset(dk->indices + entry_offset, 0xff, entry_size * usable);

    *out = dk;
    return OK;
}


/* Allocate new dictionary */
int
numba_dict_new(NB_Dict **out, Py_ssize_t size, Py_ssize_t key_size, Py_ssize_t val_size) {
    NB_DictKeys* dk;
    int status = numba_dictkeys_new(&dk, size, key_size, val_size);
    if (status != OK) return status;

    NB_Dict *d = malloc(sizeof(NB_Dict));
    if (!d) {
        numba_dictkeys_free(dk);
        return ERR_NO_MEMORY;
    }

    d->used = 0;
    d->keys = dk;
    *out = d;
    return OK;
}

/*
Adapted from CPython lookdict_index().

Search index of hash table from offset of entry table
*/
static Py_ssize_t
lookdict_index(NB_DictKeys *dk, Py_hash_t hash, Py_ssize_t index)
{
    size_t mask = D_MASK(dk);
    size_t perturb = (size_t)hash;
    size_t i = (size_t)hash & mask;

    for (;;) {
        Py_ssize_t ix = get_index(dk, i);
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

/*

Adapted from the CPython3.7 lookdict().

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
Py_ssize_t
numba_dict_lookup(NB_Dict *d, const char *key_bytes, Py_hash_t hash, char *oldval_bytes)
{
    NB_DictKeys *dk = d->keys;
    size_t mask = D_MASK(dk);
    size_t perturb = hash;
    size_t i = (size_t)hash & mask;

    for (;;) {
        Py_ssize_t ix = get_index(dk, i);
        if (ix == DKIX_EMPTY) {
            zero_val(dk, oldval_bytes);
            return ix;
        }
        if (ix >= 0) {
            NB_DictEntry *ep = get_entry(dk, ix);
            char startkey[dk->val_size];
            if (ep->hash == hash) {
                copy_key(dk, startkey, entry_get_key(dk, ep));

                int cmp = key_equal(dk, startkey, key_bytes);
                if (cmp < 0) {
                    // error'ed in comparison
                    memset(oldval_bytes, 0, dk->val_size);
                    return DKIX_ERROR;
                }
                if (cmp > 0) {
                    // key is equal; retrieve the value.
                    copy_val(dk, oldval_bytes, entry_get_val(dk, ep));
                    return ix;
                }
            }
        }
        perturb >>= PERTURB_SHIFT;
        i = (i*5 + perturb + 1) & mask;
    }
    assert(0 && "unreachable");
}


/* Internal function to find slot for an item from its hash
   when it is known that the key is not present in the dict.

   The dict must be combined. */
static Py_ssize_t
find_empty_slot(NB_DictKeys *dk, Py_hash_t hash){
    assert(dk != NULL);

    const size_t mask = D_MASK(dk);
    size_t i = hash & mask;
    Py_ssize_t ix = get_index(dk, i);
    for (size_t perturb = hash; ix >= 0;) {
        perturb >>= PERTURB_SHIFT;
        i = (i*5 + perturb + 1) & mask;
        ix = get_index(dk, i);
    }
    return i;
}

static int
insertion_resize(NB_Dict *d)
{
    return numba_dict_resize(d, D_GROWTH_RATE(d));
}

int
numba_dict_insert(
    NB_Dict    *d,
    const char *key_bytes,
    Py_hash_t   hash,
    const char *val_bytes,
    char       *oldval_bytes
    )
{
    NB_DictKeys *dk = d->keys;

    Py_ssize_t ix = numba_dict_lookup(d, key_bytes, hash, oldval_bytes);
    if (ix == DKIX_ERROR) {
        // exception in key comparision in lookup.
        goto Fail;
    }

    if (ix == DKIX_EMPTY) {
        /* Insert into new slot */
        assert ( mem_cmp_zeros(oldval_bytes, dk->val_size) == 0 );
        if (dk->usable <= 0) {
            /* Need to resize */
            if (insertion_resize(d) != OK)
                goto Fail;
            else
                dk = d->keys;     // reload
        }
        Py_ssize_t hashpos = find_empty_slot(dk, hash);
        NB_DictEntry *ep = get_entry(dk, dk->nentries);
        set_index(dk, hashpos, dk->nentries);
        copy_val(dk, entry_get_key(dk, ep), key_bytes);
        assert ( hash != -1 );
        ep->hash = hash;
        copy_val(dk, entry_get_val(dk, ep), val_bytes);

        d->used += 1;
        dk->usable -= 1;
        dk->nentries += 1;
        assert (dk->usable >= 0);
        return OK;
    }

    assert ( mem_cmp_zeros(oldval_bytes, dk->val_size) != 0 && "lookup found previous entry");
    // Replace the previous value
    copy_val(dk, entry_get_val(dk, get_entry(dk, ix)), val_bytes);

    return OK;
Fail:
    return ERR_NO_MEMORY;
}

/*
Adapted from build_indices().
Internal routine used by dictresize() to build a hashtable of entries.
*/
void
build_indices(NB_DictKeys *keys, Py_ssize_t n) {
    size_t mask = (size_t)D_MASK(keys);
    for (Py_ssize_t ix = 0; ix != n; ix++) {
        Py_hash_t hash = get_entry(keys, ix)->hash;
        size_t i = hash & mask;
        for (size_t perturb = hash; get_index(keys, i) != DKIX_EMPTY;) {
            perturb >>= PERTURB_SHIFT;
            i = mask & (i*5 + perturb + 1);
        }
        set_index(keys, i, ix);
    }

}

/*

Adapted from CPython dictresize().

Restructure the table by allocating a new table and reinserting all
items again.  When entries have been deleted, the new table may
actually be smaller than the old one.
If a table is split (its keys and hashes are shared, its values are not),
then the values are temporarily copied into the table, it is resized as
a combined table, then the me_value slots in the old table are NULLed out.
After resizing a table is always combined,
but can be resplit by make_keys_shared().
*/
int
numba_dict_resize(NB_Dict *d, Py_ssize_t minsize) {
    Py_ssize_t newsize, numentries;
    NB_DictKeys *oldkeys;

    /* Find the smallest table size > minused. */
    for (newsize = D_MINSIZE;
         newsize < minsize && newsize > 0;
         newsize <<= 1)
        ;
    if (newsize <= 0) {
        return ERR_NO_MEMORY;
    }
    oldkeys = d->keys;

    /* NOTE: Current odict checks mp->ma_keys to detect resize happen.
     * So we can't reuse oldkeys even if oldkeys->dk_size == newsize.
     * TODO: Try reusing oldkeys when reimplement odict.
     */

    /* Allocate a new table. */
    int status = numba_dictkeys_new(
        &d->keys, newsize, oldkeys->key_size, oldkeys->val_size
    );
    if (status != OK) {
        d->keys = oldkeys;
        return status;
    }
    // New table must be large enough.
    assert(d->keys->usable >= d->used);

    numentries = d->used;

    if (oldkeys->nentries == numentries) {
        NB_DictEntry *oldentries, *newentries;

        oldentries = get_entry(oldkeys, 0);
        newentries = get_entry(d->keys, 0);
        memcpy(newentries, oldentries, numentries * oldkeys->entry_size);
    }
    else {
        size_t epi = 0;
        // NB_DictEntry *ep;
        for (Py_ssize_t i=0; i<numentries; ++i) {

            /*
                ep->hash == (-1) hash means it is empty

                Here, we skip until a non empty entry is encountered.
            */
            while( get_entry(oldkeys, epi)->hash == DKIX_EMPTY ) {
                assert( mem_cmp_zeros(entry_get_val(oldkeys, get_entry(oldkeys, epi)), oldkeys->val_size) == 0 );
                epi += 1;
            }
            memcpy(
                get_entry(d->keys, i),
                get_entry(oldkeys, epi),
                oldkeys->entry_size
            );
            epi += 1;

        }

    }
    numba_dictkeys_free(oldkeys);

    build_indices(d->keys, numentries);
    d->keys->usable = numentries;
    d->keys->nentries = numentries;
    return OK;
}

/*
    Adapted from CPython delitem_common
 */
int
numba_dict_delitem(NB_Dict *d, Py_hash_t hash, Py_ssize_t ix, char *oldval_bytes)
{
    NB_DictEntry *ep;
    NB_DictKeys *dk = d->keys;
    char oldkey_bytes[dk->key_size];

    Py_ssize_t hashpos = lookdict_index(dk, hash, ix);
    assert(hashpos >= 0);

    d->used -= 1;
    ep = get_entry(dk, ix);
    set_index(dk, hashpos, DKIX_DUMMY);

    copy_key(dk, oldkey_bytes, entry_get_key(dk, ep));
    zero_key(dk, entry_get_key(dk, ep));
    zero_val(dk, entry_get_val(dk, ep));
    ep->hash = DKIX_EMPTY; // to mark it as empty;

    return OK;
}

void
numba_dict_dump_keys(NB_Dict *d) {
    Py_ssize_t i, j;
    Py_ssize_t size, n;
    NB_DictEntry *ep;
    NB_DictKeys *dk = d->keys;

    n = d->used;
    size = dk->nentries;

    printf("Key dump\n");

    for (i = 0, j = 0; i < size; i++) {
        ep = get_entry(dk, i);
        if (ep->hash != -1) {
            printf("  key=%s hash=%zu value=%s\n", entry_get_key(dk, ep), ep->hash, entry_get_val(dk, ep));
            j++;
        }
    }
    printf("j = %zd; n = %zd\n", j, n);
    assert(j == n);
}


size_t
numba_dict_iter_sizeof() {
    return sizeof(NB_DictIter);
}

void
numba_dict_iter(NB_DictIter *it, NB_Dict *d) {
    it->parent = d;
    it->parent_keys = d->keys;
    it->size = d->used;
    it->pos = 0;
}

int
numba_dict_iter_next(NB_DictIter *it, const char **key_ptr, const char **val_ptr) {
    /* Detect dictionary mutation during iteration */
    if (it->parent->keys != it->parent_keys ||
        it->parent->used != it->size) {
        return ERR_DICT_MUTATED;
    }
    NB_DictKeys *dk = it->parent_keys;
    while ( it->pos < dk->nentries ) {
        NB_DictEntry *ep = get_entry(dk, it->pos++);
        if ( ep->hash != DKIX_EMPTY ) {
            *key_ptr = entry_get_key(dk, ep);
            *val_ptr = entry_get_val(dk, ep);
            return OK;
        }
    }
    return ERR_ITER_EXHAUSTED;
}

int
numba_dict_insert_ez(
    NB_Dict    *d,
    const char *key_bytes,
    Py_hash_t   hash,
    const char *val_bytes
    )
{
    char old[d->keys->val_size];
    return numba_dict_insert(d, key_bytes, hash, val_bytes, old);
}

int
numba_dict_delitem_ez(NB_Dict *d, Py_hash_t hash, Py_ssize_t ix)
{
    char oldval_bytes[d->keys->key_size];
    return numba_dict_delitem(d, hash, ix, oldval_bytes);
}


int
numba_dict_new_minsize(NB_Dict **out, Py_ssize_t key_size, Py_ssize_t val_size)
{
    return numba_dict_new(out, D_MINSIZE, key_size, val_size);
}


#define CHECK(CASE) {                                                   \
    if ( !(CASE) ) {                                                    \
        printf("'%s' failed file %s:%d\n", #CASE, __FILE__, __LINE__);   \
        return 1;                                                       \
    }                                                                   \
}

int
numba_test_dict(void) {
    puts("test_dict");

    NB_Dict *d;
    int status;

    status = numba_dict_new(&d, D_MINSIZE, 4, 8);
    CHECK(status == OK);
    CHECK(d->keys->size == D_MINSIZE);
    CHECK(d->keys->key_size == 4);
    CHECK(d->keys->val_size == 8);
    CHECK(ix_size(d->keys->size) == 1);
    printf("align(index_size * size) = %zd\n", align(ix_size(d->keys->size) * d->keys->size));

    printf("d %p\n", d);
    printf("d->usable = %zu\n", d->keys->usable);
    printf("d[0] 0x%zx\n", (char*)get_entry(d->keys, 0) - (char*)d->keys);
    CHECK ((char*)get_entry(d->keys, 0) - (char*)d->keys->indices == d->keys->entry_offset);
    printf("d[1] 0x%zx\n", (char*)get_entry(d->keys, 1) - (char*)d->keys);
    CHECK ((char*)get_entry(d->keys, 1) - (char*)d->keys->indices == d->keys->entry_offset + d->keys->entry_size);

    char got_value[d->keys->val_size];
    Py_ssize_t ix = numba_dict_lookup(d, "bef", 0xbeef, got_value);
    printf("ix = %zd\n", ix);
    CHECK (ix == DKIX_EMPTY);

    // insert 1st key
    status = numba_dict_insert(d, "bef", 0xbeef, "1234567", got_value);
    CHECK (status == OK);
    CHECK (d->used == 1);

    // insert same key
    status = numba_dict_insert(d, "bef", 0xbeef, "1234567", got_value);
    CHECK (status == OK);
    printf("got_value %s\n", got_value);
    CHECK (d->used == 1);

    // insert 2nd key
    status = numba_dict_insert(d, "beg", 0xbeef, "1234568", got_value);
    CHECK (status == OK);
    CHECK (d->used == 2);

    // insert 3rd key
    status = numba_dict_insert(d, "beh", 0xcafe, "1234569", got_value);
    CHECK (status == OK);
    CHECK (d->used == 3);

    // replace key "bef"'s value
    status = numba_dict_insert(d, "bef", 0xbeef, "7654321", got_value);
    CHECK (status == OK);
    CHECK (d->used == 3);


    // insert 4th key
    status = numba_dict_insert(d, "bei", 0xcafe, "0_0_0_1", got_value);
    CHECK (status == OK);
    CHECK (d->used == 4);

    // insert 5th key
    status = numba_dict_insert(d, "bej", 0xcafe, "0_0_0_2", got_value);
    CHECK (status == OK);
    CHECK (d->used == 5);

    // insert 5th key & triggers resize
    status = numba_dict_insert(d, "bek", 0xcafe, "0_0_0_3", got_value);
    CHECK (status == OK);
    CHECK (d->used == 6);

    // Dump
    numba_dict_dump_keys(d);

    // Make sure everything are still in there
    ix = numba_dict_lookup(d, "bef", 0xbeef, got_value);
    CHECK (ix >= 0);
    CHECK (memcpy(got_value, "7654321", d->keys->val_size));

    ix = numba_dict_lookup(d, "beg", 0xbeef, got_value);
    CHECK (ix >= 0);
    CHECK (memcpy(got_value, "1234567", d->keys->val_size));

    ix = numba_dict_lookup(d, "beh", 0xcafe, got_value);
    printf("ix = %zd\n", ix);
    CHECK (ix >= 0);
    CHECK (memcpy(got_value, "1234569", d->keys->val_size));

    ix = numba_dict_lookup(d, "bei", 0xcafe, got_value);
    CHECK (ix >= 0);
    CHECK (memcpy(got_value, "0_0_0_1", d->keys->val_size));

    ix = numba_dict_lookup(d, "bej", 0xcafe, got_value);
    CHECK (ix >= 0);
    CHECK (memcpy(got_value, "0_0_0_2", d->keys->val_size));

    ix = numba_dict_lookup(d, "bek", 0xcafe, got_value);
    CHECK (ix >= 0);
    CHECK (memcpy(got_value, "0_0_0_3", d->keys->val_size));

    // Test delete
    ix = numba_dict_lookup(d, "beg", 0xbeef, got_value);
    status = numba_dict_delitem(d, 0xbeef, ix, got_value);
    CHECK (status == OK);

    ix = numba_dict_lookup(d, "beg", 0xbeef, got_value);
    CHECK (ix == DKIX_EMPTY); // not found

    ix = numba_dict_lookup(d, "bef", 0xbeef, got_value);
    CHECK (ix >= 0);
    ix = numba_dict_lookup(d, "beh", 0xcafe, got_value);
    CHECK (ix >= 0);

    numba_dict_free(d);
    return 0;

}

#undef CHECK