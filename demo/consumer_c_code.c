#include <Python.h>
#include <customslots.h>
#include "thestandard.h"

double call_func(PyObject *obj){
  PyCustomSlot *slot = PyCustomSlots_Find(obj, EXTENSIBLETYPE_DOUBLE_FUNC_SLOT, 0);
  double (*funcptr)(double);
  if (PY_CUSTOMSLOTS_UNLIKELY(slot == 0)) return 0.0;
  funcptr = slot->data;
  return (*funcptr)(3.0);
}
