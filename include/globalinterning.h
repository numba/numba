#ifndef Py_GLOBAL_INTERN_H
#define Py_GLOBAL_INTERN_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>

#include "interning.h"

static const char *_table_name = "_global_table_v1";
static intern_table_t _global_table = NULL;

/* Interning API */
/* Uses functions so we can get the address (and make it
   accessible from FFIs) */

/* Get an interned pointer to a key (a string).
   Returns NULL on error with an exception set. */
static const char *
PyIntern_AddKey(const char *key)
{
  if (_global_table == NULL) {
    PyErr_SetString(PyExc_AssertionError,
                    "Intern table not set, did you call PyIntern_Initialize()?");
    return NULL;
  }
  return intern_key(_global_table, key);
}

/* Intialize global interning table */
static int
PyIntern_Initialize(void) {
  PyObject *module = NULL;
  PyObject *table = NULL;
  int retval;

  if (_global_table != NULL) {
    return 0;
  }

  module = PyImport_AddModule("_global_interning"); /* borrowed ref */
  if (!module)
    goto bad;

  if (PyObject_HasAttrString(module, _table_name)) {
    table = PyObject_GetAttrString(module, _table_name);
    if (!table)
      goto bad;

    if (!PyDict_Check(table))
      PyErr_SetString(PyExc_TypeError, "Intern table is not a dict");
  } else {
    /* not found; create it */
    table = (PyObject *) intern_create_table();
    if (table == NULL)
      goto bad;

    if (PyObject_SetAttrString(module, _table_name, table) < 0)
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
  Py_XDECREF(table);
  return retval;
}


#ifdef __cplusplus
}
#endif
#endif /* !Py_GLOBAL_INTERN_H */
