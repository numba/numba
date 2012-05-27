#ifndef Py_EXTENSIBLETYPE_H
#define Py_EXTENSIBLETYPE_H
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
  /* Test for GCC > 2.95 */
  #if __GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95))
    #define likely(x)   __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
  #else /* __GNUC__ > 2 ... */
    #define likely(x)   (x)
    #define unlikely(x) (x)
  #endif /* __GNUC__ > 2 ... */
#else /* __GNUC__ */
  #define likely(x)   (x)
  #define unlikely(x) (x)
#endif /* __GNUC__ */


#include <Python.h>
#include <structmember.h>

typedef struct {
    unsigned long id;
    void *data;
} PyCustomSlot;

typedef struct {
  PyHeapTypeObject etp_base;
  Py_ssize_t etp_count; /* length of tpe_custom_slots array */
  PyCustomSlot *etp_custom_slots;
} PyHeapExtensibleTypeObject;



static PyTypeObject *PyExtensibleType_TypePtr = NULL;

#define PyCustomSlots_Init PyExtensibleType_Import

#define PyCustomSlots_Check(obj) \
  ((obj)->ob_type->ob_type == PyExtensibleType_TypePtr)

#define PyCustomSlots_Count(obj) \
  (((PyHeapExtensibleTypeObject*)(obj)->ob_type)->etp_count)

#define PyCustomSlots_Table(obj) \
  (((PyHeapExtensibleTypeObject*)(obj)->ob_type)->etp_custom_slots)

static PyCustomSlot *PyCustomSlots_Find(PyObject *obj,
                                        unsigned long id,
                                        unsigned long mask) {
  PyCustomSlot *entries;
  Py_ssize_t i;
  /* We unroll and make hitting the first slot likely(); this saved
     about 2 cycles on the test system with gcc 4.6.3, -O2 */
  if (likely(PyCustomSlots_Check(obj))) {
    if (likely(PyCustomSlots_Count(obj) > 0)) {
      entries = PyCustomSlots_Table(obj);
      if (likely((entries[0].id & mask) == id)) {
        return &entries[0];
      } else {
        for (i = 1; i != PyCustomSlots_Count(obj); ++i) {
          if ((entries[i].id & mask) == id) {
            return &entries[i];
          }
        }
      }
    }
  }
  return 0;
}


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

static PyTypeObject *
PyExtensibleType_Import(void) {
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

  if (PyExtensibleType_TypePtr != 0) {
    return PyExtensibleType_TypePtr;
  }
  
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

  extensibletype = PyDict_GetItemString(d, "extensibletype-v1");
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
    if (PyType_Ready(&_PyExtensibleType_Type_Candidate) < 0) goto bad;
    if (PyDict_SetItemString(d, "extensibletype-v1", 
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
