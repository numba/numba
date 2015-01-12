#ifndef NUMBA_PY_MODULE_H_
#define NUMBA_PY_MODULE_H_

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


#if (PY_MAJOR_VERSION >= 3)
    #define PyString_AsString PyUnicode_AsUTF8
    #define PyString_Check PyUnicode_Check
    #define PyString_FromString PyUnicode_FromString
    #define PyString_InternFromString PyUnicode_InternFromString
    #define PyInt_Type PyLong_Type
#endif

#endif /* NUMBA_PY_MODULE_H_ */

