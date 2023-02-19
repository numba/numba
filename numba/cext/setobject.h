/* Adapted from CPython3.11 Include/setobject.h
 *
 * The exact commit-id of the relevant file is:
 *
 * https://github.com/python/cpython/blob/4b8d2a1b40b88e4c658b3f5f450c146c78f2e6bd/Include/setobject.h
 *
 * WARNING:
 * Most interfaces listed here are exported (global), but they are not
 * supported, stable, or part of Numba's public API. These interfaces and their
 * underlying implementations may be changed or removed in future without
 * notice.
 * */


#ifndef NUMBA_SET_H
#define NUMBA_SET_H

#include "Python.h"
#include "cext.h"

#define SET_MINSIZE 8

typedef void (*set_refcount_op_t)(const void*);

typedef struct {
    set_refcount_op_t       key_incref;
    set_refcount_op_t       key_decref;
} set_type_based_methods_table;

typedef struct {
    /* Uses Py_ssize_t instead of Py_hash_t to guarantee word size alignment */
    Py_ssize_t  hash;
    char        *key;

} NB_SetEntry;

typedef struct {
    Py_ssize_t size;            /* Size of the active hash table*/
    Py_ssize_t filled;          /* Number active and dummy entries*/
    Py_ssize_t used;            /* Number active entries */

    Py_ssize_t key_size;        /* key_size is the sizeof key type */

    /* The table contains mask + 1 slots, and that's a power of 2.
     * We store the mask instead of the size because the mask is more
     * frequently needed.
     */
    Py_ssize_t mask;

    /* method table for type-dependent operations */
    set_type_based_methods_table methods;
    /* The table points to a fixed-size smalltable for small tables
     * or to additional malloc'ed memory for bigger tables.
     * The table pointer is never NULL which saves us from repeated
     * runtime null-tests.
     */
    NB_SetEntry *table;

    /* Placeholder for table resizing, initally same as table */
    NB_SetEntry smalltable[SET_MINSIZE];
} NB_Set;

/***** Set iterator type ***********************************************/

typedef struct {
    /* parent set */
    NB_Set        *parent;
    /* parent set entry object */
    NB_SetEntry     *table;
    /* number of keys in the set being iterated */
    Py_ssize_t       num_keys;
    /* hash table size */
    Py_ssize_t       table_size;
    /* iterator position; indicates the next position to read */
    Py_ssize_t       pos;
} NB_SetIter;


NUMBA_EXPORT_FUNC(int)          /* A test function for sets */
numba_test_set(void);           /* Returns 0 for OK; 1 for failure. */

/* Allocate a new set
Parameters
- NB_Set **out
    Output for the new set.
- Py_ssize_t size
    Hashtable size. Must be power of two.
- Py_ssize_t key_size
    Size of a key entry.
*/
NUMBA_EXPORT_FUNC(int)
numba_set_new(NB_Set **out, Py_ssize_t key_size, Py_ssize_t size);

/* Free a set */
NUMBA_GLOBAL_FUNC(void)
numba_set_free(NB_Set *setp);

/* Returns length of a set */
NUMBA_EXPORT_FUNC(Py_ssize_t) 
numba_set_length(NB_Set *setp);

/* Set the method table for type specific operations */
NUMBA_EXPORT_FUNC(void)
numba_set_set_method_table(NB_Set *setp, set_type_based_methods_table *methods);

/* Add key to the set

Parameters
- NB_Set *setp
    The set object.
- const char *key
    The key as a byte buffer.
- Py_hash_t hash
    The precomputed hash of key.
Returns
- < 0 for error
- 0 for ok
*/
NUMBA_EXPORT_FUNC(int)
numba_set_add(NB_Set *setp, char *key, Py_ssize_t hash);

/* Check if a given key exists within the set.

Parameters
- NB_Set *setp
    The set object.
- char *key
    The key as a byte buffer.
- Py_hash_t hash
    The precomputed hash of the key.
Returns
- 0 for not present
- 1 for present
*/
NUMBA_EXPORT_FUNC(int) 
numba_set_contains(NB_Set *setp, char *key, Py_ssize_t hash);

/* Discard an entry from the set
Parameters
- NB_Set *setp
    The set
- const char *key
    The key as a byte buffer.
- Py_hash_t hash
    Precomputed hash of the key to be deleted
Returns
- < 0 for error
- 0 for ok
*/
NUMBA_EXPORT_FUNC(int)
numba_set_discard(NB_Set *setp, char *key, Py_hash_t hash);


/* Remove an item from the set
Parameters
- NB_set *setp
    The set
- char *key_bytes
    Output. The key as a byte buffer
*/
NUMBA_EXPORT_FUNC(int)
numba_set_popitem(NB_Set *setp, char *key_bytes);

/* Returns the sizeof a set iterator
*/
NUMBA_EXPORT_FUNC(size_t)
numba_set_iter_sizeof(void);

/* Fill a NB_SetIter for a set to begin iteration
Parameters
- NB_SetIter *it
    Output.  Must points to memory of size at least `numba_set_iter_sizeof()`.
- NB_Set *setp
    The set to be iterated.
*/
NUMBA_EXPORT_FUNC(void)
numba_set_iter(NB_SetIter *it, NB_Set *setp);

/* Advance the iterator
Parameters
- NB_SetIter *it
    The iterator
- const char **key_ptr
    Output pointer for the key.  Points to data in the set.

Returns
- 0 for success; valid key_ptr
- ERR_ITER_EXHAUSTED for end of iterator.
- ERR_SET_MUTATED for detected set mutation.
*/
NUMBA_EXPORT_FUNC(int)
numba_set_iter_next(NB_SetIter *it, const char **key_ptr);


NUMBA_EXPORT_FUNC(void)
numba_set_dump(NB_Set *setp);

#endif
