#include <Python.h>
#include <extensibletype.h>
#include "thestandard.h"

void init_c_code(void) {
  PyExtensibleType_Import();
}

double call_func(PyObject *obj){
  PyCustomSlot *slot = PyCustomSlots_Find(obj, EXTENSIBLETYPE_DOUBLE_FUNC_SLOT, 0xffffff00);
  double (*funcptr)(double);
  if (unlikely(slot == 0)) return 0.0;
  funcptr = slot->data;
  return (*funcptr)(3.0);
}
