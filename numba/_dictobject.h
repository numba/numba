/* Adapted from CPython3.7 Objects/dict-common.h */

#ifndef NUMBA_DICT_COMMON_H
#define NUMBA_DICT_COMMON_H

typedef struct {
    Py_hash_t   hash;
    char        keyvalue[];
} NB_DictEntry;

typedef struct {
   /* hash table size */
    Py_ssize_t      size;
    /* Usable size of the hash table.
       Also, size of the entries */
    Py_ssize_t      usable;
    /* hash table used entries */
    Py_ssize_t      nentries;
    /* Entry info
        - key_size is the sizeof key type
        - val_size is the sizeof value type
        - entry_size is key_size + val_size + alignment
    */
    Py_ssize_t      key_size, val_size, entry_size;
    /* Byte offset from indices to the first entry. */
    Py_ssize_t      entry_offset;

    /* hash table */
    char            indices[];
} NB_DictKeys;


typedef struct {
    /* num of elements in the hashtable */
    Py_ssize_t        used;
    NB_DictKeys      *keys;
} NB_Dict;


typedef struct {
    /* parent dictionary */
    NB_Dict         *parent;
    /* parent keys object */
    NB_DictKeys     *parent_keys;
    /* dict size */
    Py_ssize_t       size;
    /* iterator position; indicates the next position to read */
    Py_ssize_t       pos;
} NB_DictIter;



/* A test function for the dict */
NUMBA_EXPORT_FUNC(int)
numba_test_dict(void);

/* Allocate a new dict */
NUMBA_EXPORT_FUNC(int)
numba_dict_new(NB_Dict **out, Py_ssize_t size, Py_ssize_t key_size, Py_ssize_t val_size);

/* Free a dict */
NUMBA_EXPORT_FUNC(void)
numba_dict_free(NB_Dict *d);

/* Returns length of a dict */
NUMBA_EXPORT_FUNC(Py_ssize_t)
numba_dict_length(NB_Dict *d);

/* Allocates a new dict at the minimal size */
NUMBA_EXPORT_FUNC(int)
numba_dict_new_minsize(NB_Dict **out, Py_ssize_t key_size, Py_ssize_t val_size);

/* Lookup a key */
NUMBA_EXPORT_FUNC(Py_ssize_t)
numba_dict_lookup(NB_Dict *d, const char *key_bytes, Py_hash_t hash, char *oldval_bytes);

/* Resize the dict */
NUMBA_EXPORT_FUNC(int)
numba_dict_resize(NB_Dict *d, Py_ssize_t minsize);

/* Insert to the dict */
NUMBA_EXPORT_FUNC(int)
numba_dict_insert(NB_Dict *d, const char *key_bytes, Py_hash_t hash, const char *val_bytes, char *oldval_bytes);

/* Same as numba_dict_insert() but oldval_bytes is not needed */
NUMBA_EXPORT_FUNC(int)
numba_dict_insert_ez(NB_Dict *d, const char *key_bytes, Py_hash_t hash, const char *val_bytes);

/* Delete an entry from the dict */
NUMBA_EXPORT_FUNC(int)
numba_dict_delitem(NB_Dict *d, Py_hash_t hash, Py_ssize_t ix, char *oldval_bytes);

/* Same as numba_dict_delitem() but oldval_bytes is not needed */
NUMBA_EXPORT_FUNC(int)
numba_dict_delitem_ez(NB_Dict *d, Py_hash_t hash, Py_ssize_t ix);

/* Returns the sizeof a dictionary iterator */
NUMBA_EXPORT_FUNC(size_t)
numba_dict_iter_sizeof(void);

/* Fill a NB_DictIter for a dictionary to begin iteration */
NUMBA_EXPORT_FUNC(void)
numba_dict_iter(NB_DictIter *it, NB_Dict *d);

/* Advance the iterator */
NUMBA_EXPORT_FUNC(int)
numba_dict_iter_next(NB_DictIter *it, const char **key_ptr, const char **val_ptr);


#endif