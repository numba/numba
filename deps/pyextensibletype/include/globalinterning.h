#ifndef Py_GLOBAL_INTERN_H
#define Py_GLOBAL_INTERN_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>

#include "interning.h"

static const char *_table_name = "_global_table_v1";
static intern_table_t _global_intern_table;
static intern_table_t *_global_table = NULL;

/* Interning API */

/* Uses functions so we can get the address (and make it
   accessible from FFIs) */

/* Get a unique prehash for a signature string.
   Returns 0 on error with an exception set ('except? 0'). */
static uint64_t
PyIntern_AddKey(const char *key)
{
  if (_global_table == NULL) {
    PyErr_SetString(PyExc_AssertionError,
                    "Intern table not set, did you call PyIntern_Initialize()?");
    return 0;
  }

  return intern_key(_global_table, key);
}

static PyObject *
capsule_create(void *p, const char *sig)
{
  PyObject *capsule;

#if PY_VERSION_HEX >= 0x02070000 && !(PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 0)
  capsule = PyCapsule_New(p, sig, NULL);
#else
  capsule = PyCObject_FromVoidPtr(p, NULL);
#endif

  return capsule;
}

static void *
capsule_getpointer(PyObject *capsule, const char *sig)
{
  void *cobj;

#if PY_VERSION_HEX >= 0x02070000 && !(PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 0)
  cobj = PyCapsule_GetPointer(capsule, sig);
#else
  cobj = PyCObject_AsVoidPtr(capsule);
#endif

  return cobj;
}

/* Intialize global interning table */
static int
PyIntern_Initialize(void) {
  PyObject *module = NULL;
  intern_table_t *table = NULL;
  PyObject *capsule = NULL;
  int retval;

  if (_global_table != NULL) {
    return 0;
  }

  module = PyImport_AddModule("_global_interning"); /* borrowed ref */
  if (!module)
    goto bad;

  if (PyObject_HasAttrString(module, _table_name)) {
    capsule = PyObject_GetAttrString(module, _table_name);
    if (!capsule)
      goto bad;

    table = capsule_getpointer(capsule, "_intern_table");
    if (!table)
      goto bad;
  } else {
    /* not found; create it */
    table = intern_create_table(&_global_intern_table);
    if (table == NULL)
      goto bad;

    capsule = capsule_create(table, "_intern_table");
    if (PyObject_SetAttrString(module, _table_name, capsule) < 0)
      goto bad;
  }

  /* Initialize the global variable used in macros */
  _global_table = table;

  retval = 0;
  goto ret;
bad:
  retval = -1;
ret:
  /* module is borrowed */
  Py_XDECREF(capsule);
  return retval;
}


#ifdef __cplusplus
}
#endif
#endif /* !Py_GLOBAL_INTERN_H */
