#include "structmember.h"
#include "thestandard.h"
#include "extensibletype.h"

#if PY_VERSION_HEX < 0x02050000
  #define NAMESTR(n) ((char *)(n))
#else
  #define NAMESTR(n) (n)
#endif

double func(double x) {
  return x * x;
}


static PyObject *Provider_new(PyTypeObject *t, PyObject *a, PyObject *k) {
  PyObject *o = (*t->tp_alloc)(t, 0);
  if (!o) return 0;
  return o;
}

static void Provider_dealloc(PyObject *o) {
  (*Py_TYPE(o)->tp_free)(o);
}


typedef struct {
  PyObject_HEAD
} Provider_Object;

PyCustomSlot my_custom_slots[1] = {
  {EXTENSIBLETYPE_DOUBLE_FUNC_SLOT, func}
};

PyHeapExtensibleTypeObject Provider_Type = {
  /* PyHeapTypeObject etp_heaptype */
  {

    /* PyTypeObject ht_type */
    {
      PyVarObject_HEAD_INIT(0, 0)
      NAMESTR("providertype"), /*tp_name*/
      sizeof(Provider_Object),         /* tp_basicsize */
      0,                        /* tp_itemsize */
      &Provider_dealloc, /*tp_dealloc*/
      0, /*tp_print*/
      0, /*tp_getattr*/
      0, /*tp_setattr*/
#if PY_MAJOR_VERSION < 3
      0, /*tp_compare*/
#else
      0, /*reserved*/
#endif
      0, /*tp_repr*/
      &Provider_Type.etp_heaptype.as_number, /*tp_as_number*/
      &Provider_Type.etp_heaptype.as_sequence, /*tp_as_sequence*/
      &Provider_Type.etp_heaptype.as_mapping, /*tp_as_mapping*/
      0, /*tp_hash*/
      0, /*tp_call*/
      0, /*tp_str*/
      0, /*tp_getattro*/
      0, /*tp_setattro*/
      &Provider_Type.etp_heaptype.as_buffer, /*tp_as_buffer*/
      Py_TPFLAGS_DEFAULT|Py_TPFLAGS_CHECKTYPES|Py_TPFLAGS_BASETYPE, /*tp_flags*/
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
      0, /*tp_base*/
      0, /*tp_dict*/
      0, /*tp_descr_get*/
      0, /*tp_descr_set*/
      0, /*tp_dictoffset*/
      0, /*tp_init*/
      0, /*tp_alloc*/
      &Provider_new, /*tp_new*/
      0, /*tp_free*/
      0, /*tp_is_gc*/
      0, /*tp_bases*/
      0, /*tp_mro*/
      0, /*tp_cache*/
      0, /*tp_subclasses*/
      0, /*tp_weaklist*/
      0, /*tp_del*/
#if PY_VERSION_HEX >= 0x02060000
      0 /*tp_version_tag*/
#endif
    },

    /* PyNumberMethods as_number */
    {
      0, /*nb_add*/
      0, /*nb_subtract*/
      0, /*nb_multiply*/
      0, /*nb_divide*/
      0, /*nb_remainder*/
      0, /*nb_divmod*/
      0, /*nb_power*/
      0, /*nb_negative*/
      0, /*nb_positive*/
      0, /*nb_absolute*/
      0, /*nb_nonzero*/
      0, /*nb_invert*/
      0, /*nb_lshift*/
      0, /*nb_rshift*/
      0, /*nb_and*/
      0, /*nb_xor*/
      0, /*nb_or*/
      0, /*nb_coerce*/
      0, /*nb_int*/
      0, /*nb_long*/
      0, /*nb_float*/
      0, /*nb_oct*/
      0, /*nb_hex*/
      0, /*nb_inplace_add*/
      0, /*nb_inplace_subtract*/
      0, /*nb_inplace_multiply*/
      0, /*nb_inplace_divide*/
      0, /*nb_inplace_remainder*/
      0, /*nb_inplace_power*/
      0, /*nb_inplace_lshift*/
      0, /*nb_inplace_rshift*/
      0, /*nb_inplace_and*/
      0, /*nb_inplace_xor*/
      0, /*nb_inplace_or*/
      0, /*nb_floor_divide*/
      0, /*nb_true_divide*/
      0, /*nb_inplace_floor_divide*/
      0, /*nb_inplace_true_divide*/
#if PY_VERSION_HEX >= 0x02050000
      0, /*nb_index*/
#endif
    },

    /* PyMappingMethods as_mapping */
    {
      0, /*mp_length*/
      0, /*mp_subscript*/
      0, /*mp_ass_subscript*/
    },

    /* PySequenceMethods as_sequence */
    {
      0, /*sq_length*/
      0, /*sq_concat*/
      0, /*sq_repeat*/
      0, /*sq_item*/
      0, /*sq_slice*/
      0, /*sq_ass_item*/
      0, /*sq_ass_slice*/
      0, /*sq_contains*/
      0, /*sq_inplace_concat*/
      0, /*sq_inplace_repeat*/
    },

    /* PyBufferProcs as_buffer */
    {
      0, /*bf_getreadbuffer*/
      0, /*bf_getwritebuffer*/
      0, /*bf_getsegcount*/
      0, /*bf_getcharbuffer*/
#if PY_VERSION_HEX >= 0x02060000
      0, /*bf_getbuffer*/
      0, /*bf_releasebuffer*/
#endif
    },

    0, /* ht_name */
    0 /* ht_slots */
    
  }, /* end of PyHeapTypeObject */

  1, /* etp_custom_slot_count */
  my_custom_slots /* etp_custom_slot_table */

};

int ProviderType_Ready(void) {
  /* Set as_number, as_buffer etc. to 0; these could of course be
     explicitly initialized too */
  if (PyExtensibleType_Ready(&Provider_Type, 1) < 0) {
    return -1;
  }
  return 0;
}
