#ifndef Py_EXTENSIBLETYPE_H
#define Py_EXTENSIBLETYPE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <structmember.h>

typedef struct {
    unsigned long id;
    void *data;
} PyExtensibleTypeObjectEntry;

typedef struct {
  PyHeapTypeObject etp_base;
  Py_ssize_t etp_count; /* length of tpe_entries array */
  PyExtensibleTypeObjectEntry *etp_custom_slots;
} PyHeapExtensibleTypeObject;

static PyTypeObject PyExtensibleType_Type = {
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
  0, /*tp_new*/
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


static PyTypeObject *
PyExtensibleType_GetMetaClass(void) {
  if (PyType_Ready(&PyExtensibleType_Type) < 0) {
    return NULL;
  }
  return &PyExtensibleType_Type;
}



#ifdef __cplusplus
}
#endif
#endif /* !Py_EXTENSIBLETYPE_H */
