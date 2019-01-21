/* Adapted from CPython3.7 Objects/dict-common.h */

#ifndef Py_DICT_COMMON_H
#define Py_DICT_COMMON_H


typedef struct _NumbaKeyObject{ void * ptr; } NumbaKeyObject;
typedef struct _NumbaValObject{ void * ptr; } NumbaValObject;

typedef struct {
    /* Cached hash code of me_key. */
    Py_hash_t me_hash;
    size_t    me_is_used;
    NumbaKeyObject me_key;
    NumbaValObject me_value; /* This field is only meaningful for combined tables */
} NumbaDictKeyEntry;


struct _dictkeysobject;


/* The following structure is adapted from PyDictObject from
   cpython/dictobject.h */
typedef struct {

    /* Number of items in the dictionary */
    Py_ssize_t ma_used;

    /* Dictionary version: globally unique, value change each time
       the dictionary is modified */
    uint64_t ma_version_tag;

    struct _dictkeysobject *ma_keys;


} NumbaDictObject;


/* dict_lookup_func() returns index of entry which can be used like DK_ENTRIES(dk)[index].
 * -1 when no entry found, -3 when compare raises error.
 */
typedef Py_ssize_t (*dict_lookup_func)
    (NumbaDictObject *mp, NumbaKeyObject key, Py_hash_t hash, NumbaValObject *value_addr);

#define DKIX_EMPTY (-1)
#define DKIX_DUMMY (-2)  /* Used internally */
#define DKIX_ERROR (-3)

/* See dictobject.c for actual layout of DictKeysObject */
typedef struct _dictkeysobject {
    Py_ssize_t dk_refcnt;

    /* Size of the hash table (dk_indices). It must be a power of 2. */
    Py_ssize_t dk_size;

    /* Function to lookup in the hash table (dk_indices):

       - lookdict(): general-purpose, and may return DKIX_ERROR if (and
         only if) a comparison raises an exception.

       - lookdict_unicode(): specialized to Unicode string keys, comparison of
         which can never raise an exception; that function can never return
         DKIX_ERROR.

       - lookdict_unicode_nodummy(): similar to lookdict_unicode() but further
         specialized for Unicode string keys that cannot be the <dummy> value.

       - lookdict_split(): Version of lookdict() for split tables. */
    dict_lookup_func dk_lookup;

    /* Number of usable entries in dk_entries. */
    Py_ssize_t dk_usable;

    /* Number of used entries in dk_entries. */
    Py_ssize_t dk_nentries;


    Py_ssize_t dk_value_size;

    /* Actual hash table of dk_size entries. It holds indices in dk_entries,
       or DKIX_EMPTY(-1) or DKIX_DUMMY(-2).

       Indices must be: 0 <= indice < USABLE_FRACTION(dk_size).

       The size in bytes of an indice depends on dk_size:

       - 1 byte if dk_size <= 0xff (char*)
       - 2 bytes if dk_size <= 0xffff (int16_t*)
       - 4 bytes if dk_size <= 0xffffffff (int32_t*)
       - 8 bytes otherwise (int64_t*)

       Dynamically sized, SIZEOF_VOID_P is minimum. */
    char dk_indices[];  /* char is required to avoid strict aliasing. */

    /* "NumbaDictKeyEntry dk_entries[dk_usable];" array follows:
       see the DK_ENTRIES() macro */
} NumbaDictKeysObject;


/* XXX: unknown macros */
#define _Numba_HOT_FUNCTION
#define Numba_UNREACHABLE() exit(1)

#endif