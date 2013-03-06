#ifndef Py_INTERNING_H
#define Py_INTERNING_H
#ifdef __cplusplus
extern "C" {
#endif

/* Utility for interning strings */
/* TODO: make it GIL-less and Python independent */

#include <Python.h>

#if PY_MAJOR_VERSION < 3
    #define _PyIntern_FromString PyString_FromString
    #define _PyIntern_FromStringAndSize PyString_FromStringAndSize
    #define _PyIntern_AsString PyString_AsString
#else
    #define _PyIntern_FromString PyBytes_FromString
    #define _PyIntern_FromStringAndSize PyBytes_FromStringAndSize
    #define _PyIntern_AsString PyBytes_AsString
#endif


typedef void *intern_table_t;

static intern_table_t
intern_create_table(void)
{
    /* { string -> interned_string } */
    PyObject *table = PyDict_New();
    return (intern_table_t) table;
}

static void
intern_destroy_table(intern_table_t table)
{
    Py_DECREF((PyObject *) table);
}

static const char *
_intern_key(intern_table_t table, PyObject *key)
{
    PyObject *dict = (PyObject *) table;
    PyObject *value;

    value = PyDict_GetItem(dict, key);

    if (value == NULL) {
        /* Key not in dict */
        value = key;
        PyDict_SetItem(dict, key, value);
    }

    return _PyIntern_AsString(value);
}

static const char *
intern_key(intern_table_t table, const char *key)
{
    PyObject *key_obj = _PyIntern_FromString(key);
    const char *retval;

    if (key_obj == NULL)
        return NULL;

    retval = _intern_key(table, key_obj);

    Py_DECREF(key_obj);
    return retval;
}


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNING_H */
