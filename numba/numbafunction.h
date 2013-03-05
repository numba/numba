extern size_t closure_field_offset;
extern PyTypeObject *NumbaFunctionType;
extern int NumbaFunction_init();
extern PyObject *NumbaFunction_NewEx(
                PyMethodDef *ml, PyObject *module, PyObject *code,
                PyObject *closure, void *native_func,
                PyObject *native_signature, PyObject *keep_alive);
