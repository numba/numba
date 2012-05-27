#include <Python.h>
#include <extensibletype.h>
#include "thestandard.h"

void init_c_code(void) {
  PyExtensibleType_Import();
}

double call_func(PyObject *obj){
  double (*funcptr)(double);
  funcptr = PyCustomSlots_Find(obj, EXTENSIBLETYPE_DOUBLE_FUNC_SLOT, 0xffffff00);
  if (likely(funcptr)) return funcptr(3.0);
  else return 0.0;
}
