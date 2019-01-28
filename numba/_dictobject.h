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
    Py_ssize_t      usable;
    /* hash table used entries */
    Py_ssize_t      nentries;
    /* Entry info */
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


#endif