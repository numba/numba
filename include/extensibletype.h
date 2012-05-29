#ifndef Py_EXTENSIBLETYPE_H
#define Py_EXTENSIBLETYPE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "customslots.h"

/*
The metaclass definition. Do not use directly, but instead call
PyExtensibleType_Import.
*/

static PyObject *_PyExtensibleType_new(PyTypeObject *t, PyObject *a, PyObject *k) {
  PyHeapExtensibleTypeObject* new_type, *base_type;
  PyObject *o = (*PyType_Type.tp_new)(t, a, k);
  if (!o) return 0;
  new_type = (PyHeapExtensibleTypeObject*)o;
  base_type = (PyHeapExtensibleTypeObject*)((PyTypeObject*)o)->tp_base;
  new_type->etp_count = base_type->etp_count;
  new_type->etp_custom_slots = base_type->etp_custom_slots;
  ((PyTypeObject*)new_type)->tp_flags |= PyExtensibleType_TPFLAGS_IS_EXTENSIBLE;
  return o;
}

static PyTypeObject _PyExtensibleType_Type_Candidate = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
#if PY_VERSION_HEX < 0x02050000
  (char *)"extensibletype",  /*tp_name*/
#else
  "extensibletype",  /*tp_name*/
#endif
  sizeof(PyHeapExtensibleTypeObject),         /* tp_basicsize */
  sizeof(PyMemberDef),                        /* tp_itemsize */
  0, /*tp_dealloc*/
  0, /*tp_print*/
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  #if PY_MAJOR_VERSION < 3
  0, /*tp_compare*/
  #else
  0, /*reserved*/
  #endif
  0, /*tp_repr*/
  0, /*tp_as_number*/
  0, /*tp_as_sequence*/
  0, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_TYPE_SUBCLASS, /*tp_flags*/
  0, /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  0, /*tp_methods*/
  0, /*tp_members*/
  0, /*tp_getset*/
  &PyType_Type, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  0, /*tp_init*/
  0, /*tp_alloc*/
  &_PyExtensibleType_new, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
  0, /*tp_del*/
  #if PY_VERSION_HEX >= 0x02060000
  0, /*tp_version_tag*/
  #endif
};

static PyTypeObject *PyExtensibleType_TypePtr = 0;

static PyTypeObject *
PyExtensibleType_Import(void) {
  /* Performs roughly the equivalent of:

     d = sys.modules.setdefault('_extensibletype', {})
     return d.setdefault('extensibletype', our_extensibletype);

     If another module got to sys.modules first, the
     static PyExtensibleType_Type defined above is left unused.
   */
  PyObject *module = 0;
  PyObject *extensibletype = 0;
  PyTypeObject *retval;

  if (PyExtensibleType_TypePtr != 0) {
    return PyExtensibleType_TypePtr;
  }

  module = PyImport_AddModule("_extensibletype"); /* borrowed ref */
  if (!module) goto bad;

  if (PyObject_HasAttrString(module, "extensibletype_v1")) {
    extensibletype = PyObject_GetAttrString(module, "extensibletype_v1");
    if (!extensibletype) goto bad;
    if (!PyType_Check(extensibletype) || 
        ((PyTypeObject*)extensibletype)->tp_basicsize !=
        sizeof(PyHeapExtensibleTypeObject)) {
      PyErr_SetString(PyExc_TypeError,
                      "'extensibletype' found but is wrong type or size");
    }
    retval = (PyTypeObject*)extensibletype;
  } else {
    /* not found; create it */
    if (PyType_Ready(&_PyExtensibleType_Type_Candidate) < 0) goto bad;
    if (PyObject_SetAttrString(module, "extensibletype_v1", 
        (PyObject*)&_PyExtensibleType_Type_Candidate) < 0) goto bad;
    retval = (PyTypeObject*)&_PyExtensibleType_Type_Candidate;
    Py_INCREF((PyObject*)retval);
  }

  /* Initialize the global variable used in macros */
  PyExtensibleType_TypePtr = retval;

  goto ret;
 bad:
  retval = NULL;
 ret:
  /* module is borrowed */
  Py_XDECREF(extensibletype);
  return retval;
}



#ifdef __cplusplus
}
#endif
#endif /* !Py_EXTENSIBLETYPE_H */
