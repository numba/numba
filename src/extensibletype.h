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
PyExtensibleType_Init_(void) {
  return &PyExtensibleType_Type;
}


static PyTypeObject *
PyExtensibleType_GetMetaClass(void) {
  /* Performs roughly the equivalent of:

     d = sys.modules.setdefault('_extensibletype', {})
     return d.setdefault('extensibletype', our_extensibletype);

     If another module got to sys.modules first, the
     static PyExtensibleType_Type defined above is left unused.
   */
  PyObject *sys = 0;
  PyObject *modules = 0;
  PyObject *d = 0;
  PyObject *extensibletype = 0;
  PyTypeObject *retval;
  
  sys = PyImport_ImportModule("sys");
  if (!sys) goto bad;
  modules = PyObject_GetAttrString(sys, "modules");
  if (!modules) goto bad;
  if (!PyDict_Check(modules)) {
      PyErr_SetString(PyExc_TypeError,
                      "sys.modules is not a dict");
      goto bad;
  }

  d = PyDict_GetItemString(modules, "_extensibletype");
  if (d) {
    Py_INCREF(d); /* borrowed ref */
    if (!PyDict_Check(d)) {
      PyErr_SetString(PyExc_TypeError,
                      "sys.modules['_extensibletype'] is not a dict");
      goto bad;
    }
  } else {
    d = PyDict_New();
    if (!d) goto bad;
    if (PyDict_SetItemString(modules, "_extensibletype", d) < 0) goto bad;
  }

  extensibletype = PyDict_GetItemString(d, "extensibletype");
  if (extensibletype) {
    Py_INCREF(extensibletype); /* borrowed reference */
    if (!PyType_Check(extensibletype) || 
        ((PyTypeObject*)extensibletype)->tp_basicsize !=
        sizeof(PyHeapExtensibleTypeObject)) {
      PyErr_SetString(PyExc_TypeError,
                      "'extensibletype' found but is wrong type or size");
    }
    retval = (PyTypeObject*)extensibletype;
  } else {
    /* not found; create it */
    if (PyType_Ready(&PyExtensibleType_Type) < 0) goto bad;
    if (PyDict_SetItemString(d, "extensibletype", 
                             (PyObject*)&PyExtensibleType_Type) < 0) goto bad;
    Py_INCREF((PyObject*)&PyExtensibleType_Type);
    retval = (PyTypeObject*)&PyExtensibleType_Type;
  }
  goto ret;
 bad:
  retval = NULL;
 ret:
  Py_XDECREF(sys);
  Py_XDECREF(modules);
  Py_XDECREF(d);
  Py_XDECREF(extensibletype);
  return retval;
}



#ifdef __cplusplus
}
#endif
#endif /* !Py_EXTENSIBLETYPE_H */
