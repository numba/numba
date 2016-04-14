#ifndef NUMBA_PY_MODULE_H_
#define NUMBA_PY_MODULE_H_

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <structmember.h>

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) { \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef); }
  #define MOD_INIT_EXEC(name) PyInit_##name();
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
  #define MOD_INIT_EXEC(name) init##name();
#endif


#if PY_MAJOR_VERSION >= 3
    #define PyString_AsString PyUnicode_AsUTF8
    #define PyString_Check PyUnicode_Check
    #define PyString_FromFormat PyUnicode_FromFormat
    #define PyString_FromString PyUnicode_FromString
    #define PyString_InternFromString PyUnicode_InternFromString
    #define PyInt_Type PyLong_Type
    #define PyInt_Check PyLong_Check
    #define PyInt_CheckExact PyLong_CheckExact
#else
    #define Py_hash_t long
    #define Py_uhash_t unsigned long
#endif

#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 4)
    #define PyMem_RawMalloc malloc
    #define PyMem_RawRealloc realloc
    #define PyMem_RawFree free
#endif

#ifndef Py_MIN
#define Py_MIN(x, y) (((x) > (y)) ? (y) : (x))
#endif

#ifndef Py_MAX
#define Py_MAX(x, y) (((x) < (y)) ? (y) : (x))
#endif

#endif /* NUMBA_PY_MODULE_H_ */
