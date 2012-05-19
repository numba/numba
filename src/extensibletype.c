#include "extensibletype.h"
#include "structmember.h"

#if PY_VERSION_HEX < 0x02050000
  #define NAMESTR(n) ((char *)(n))
#else
  #define NAMESTR(n) (n)
#endif

static PyTypeObject PyExtensibleType_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  NAMESTR("extensibletype"), /*tp_name*/
  sizeof(PyHeapTypeObject),                   /* tp_basicsize */
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


PyTypeObject *
PyExtensibleType_GetMetaClass(void) {
  if (PyType_Ready(&PyExtensibleType_Type) < 0) {
    return NULL;
  }
  return &PyExtensibleType_Type;
}
