#ifndef Py_INTERNING_H
#define Py_INTERNING_H
#ifdef __cplusplus
extern "C" {
#endif

/* Utility for interning strings */
/* TODO: make it GIL-less and Python independent */

#include <Python.h>
#include <stdlib.h>
#include "pstdint.h"
#include "siphash24.h"

#if PY_MAJOR_VERSION < 3
    #define _PyIntern_FromString PyString_FromString
    #define _PyIntern_FromStringAndSize PyString_FromStringAndSize
    #define _PyIntern_AsString PyString_AsString
    #define _PyIntern_Size PyString_Size
#else
    #define _PyIntern_FromString PyBytes_FromString
    #define _PyIntern_FromStringAndSize PyBytes_FromStringAndSize
    #define _PyIntern_AsString PyBytes_AsString
    #define _PyIntern_Size PyBytes_Size
#endif

/* Data types */

typedef struct _intern_table_t {
    PyObject *signature_to_key;
    PyObject *key_to_signature;
    char secrets[16*4]; /* 4 secret keys, which we try in succession */
} intern_table_t;

/* Prototypes */
static void intern_destroy_table(intern_table_t *table);

/* API */

static void
_print_secrets(intern_table_t *table)
{
    int i, j;

    for (i = 0; i < 4; i++) {
        printf("secret key[%d] = {", i);
        for (j = 0; j < 16; j += 4) {
            printf(" %-8x, ", *(int32_t *) &table->secrets[i * 16 + j]);
        }
        printf("}\n");
    }
}

/* Create an intern table from preallocated memory.
   Returns NULL on failure with an appropriate exception set. */
static intern_table_t *
intern_create_table(intern_table_t *table)
{
    int i;

    table->signature_to_key = NULL;
    table->key_to_signature = NULL;

    table->signature_to_key = PyDict_New();
    table->key_to_signature = PyDict_New();

    if (!table->signature_to_key || !table->key_to_signature)
        goto bad;

    for (i = 0; i < 16 * 4; i += 2) {
        /* Take the lower two bytes from the random value, since
               RAND_MAX is at least 2**16 */
        short randval = (short) rand(); /* TODO: use a better prng */

        table->secrets[i + 0] = ((char *) &randval)[0];
        table->secrets[i + 1] = ((char *) &randval)[1];
    }
    /* Amend this! */
    memset(&table->secrets[16*0], 0, 16);
    memset(&table->secrets[16*1], 1, 16);
    memset(&table->secrets[16*2], 2, 16);
    memset(&table->secrets[16*3], 3, 16);

    /* _print_secrets(table); */

    return table;
bad:
    intern_destroy_table(table);
    return NULL;
}

static void
intern_destroy_table(intern_table_t *table)
{
    Py_CLEAR(table->signature_to_key);
    Py_CLEAR(table->key_to_signature);
}

/*
    Update table with a prehash candidate.

    Returns -1 on error, 0 on duplicate prehash, 1 on success.
 */
static int
update_table(intern_table_t *table, PyObject *key_obj, uint64_t prehash)
{
    PyObject *value;
    int retcode;
    int result;

    /* TODO: Py_LONG_LONG may not be 64 bits... */
    #if PY_ULLONG_MAX < 0xffffffffffffffffULL
        #error "sizeof(unsigned PY_LONG_LONG) must be at least 8 bytes"
    #endif

    value = PyLong_FromUnsignedLongLong(prehash);
    if (!value)
        goto bad;

    /* See whether we already have this hash for a different signature string */
    result = PyDict_Contains(table->key_to_signature, value);
    if (result != 0) {
        if (result == -1)
            goto bad;
        else
            goto duplicate;
    }

    if (PyDict_SetItem(table->signature_to_key, key_obj, value) < 0)
        goto bad;

    if (PyDict_SetItem(table->key_to_signature, value, key_obj) < 0) {
        PyDict_DelItem(table->signature_to_key, key_obj);
        goto bad;
    }

    retcode = 1;
    goto done;

bad:
    retcode = -1;

duplicate:
    retcode = 0;

done:
    Py_XDECREF(value);
    return retcode;
}

/* Build prehash using siphash given the signature string and a secret key */
static uint64_t
_intern_build_key(PyObject *key_obj, const char *key, const char *secret)
{
    Py_ssize_t len = _PyIntern_Size(key_obj);
    uint64_t prehash;
    (void) crypto_auth((unsigned char *) &prehash,
                       (const unsigned char *) key,
                       len,
                       (const unsigned char *) secret);
    return prehash;
}

/* Make a prehash for a signature string, trying different secret keys in
   succession. */
static int
make_prehash(intern_table_t *table, PyObject *key_obj, const char *key,
             uint64_t *prehash_out)
{
    const char *secret = table->secrets;
    int tries = 0;
    uint64_t prehash;

    while (1) {
        int result;
        prehash = _intern_build_key(key_obj, key, secret);
        result = update_table(table, key_obj, prehash);
        if (result < 0) {
            goto bad;
        } else if (result == 0) {
            /* Duplicate, keep going */
            secret += 16;
            if (++tries == 4) {
                PyErr_SetString(PyExc_ValueError,
                                "Failed to create unique prehash");
                goto bad;
            }
        } else {
            /* We have a unique prehash */
            break;
        }
    }

    *prehash_out = prehash;
    return 0;
bad:
    return -1;
}

static uint64_t
_intern_key(intern_table_t *table, PyObject *key_obj, const char *key)
{
    PyObject *value;
    PyObject *tmp = NULL;
    uint64_t prehash;

    value = PyDict_GetItem(table->signature_to_key, key_obj);

    if (value == NULL) {
        /* Key not in dict */
        Py_INCREF(key_obj);
        if (make_prehash(table, key_obj, key, &prehash) < 0)
            goto bad;
    } else {
        prehash = PyLong_AsUnsignedLongLong(value);
        if (PyErr_Occurred())
            goto bad;
    }

    goto done;

bad:
    prehash = 0;

done:
    Py_XDECREF(tmp);
    return prehash;
}

/*

   Intern a signature string and return a unique prehash, to be used to
   compute the final hash in a perfect hashing vtable.

   Callers should check for errors using PyErr_Occurred() when this function
   returns 0.
*/
static uint64_t
intern_key(intern_table_t *table, const char *key)
{
    PyObject *key_obj = _PyIntern_FromString(key);
    uint64_t retval;

    if (key_obj == NULL)
        return 0;

    retval = _intern_key(table, key_obj, key);

    Py_DECREF(key_obj);
    return retval;
}


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNING_H */
