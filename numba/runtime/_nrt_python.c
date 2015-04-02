#include "_pymodule.h"
#include "nrt.h"


static
PyObject*
memsys_set_atomic_inc_dec(PyObject *self, PyObject *args) {
    PY_LONG_LONG addr_inc, addr_dec;
    if (!PyArg_ParseTuple(args, "KK", &addr_inc, &addr_dec)) {
        return NULL;
    }
    NRT_MemSys_set_atomic_inc_dec((void*)addr_inc, (void*)addr_dec);
    Py_RETURN_NONE;
}

static
PyObject*
memsys_process_defer_dtor(PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    NRT_MemSys_process_defer_dtor();
    Py_RETURN_NONE;
}


static
PyObject*
meminfo_acquire(PyObject *self, PyObject *args) {
    PY_LONG_LONG mi_addr;
    if (!PyArg_ParseTuple(args, "K", &mi_addr)) {
        return NULL;
    }
    NRT_MemInfo_acquire((MemInfo*)mi_addr);
    Py_RETURN_NONE;
}

static
PyObject*
meminfo_release(PyObject *self, PyObject *args) {
    PY_LONG_LONG mi_addr;
    PyObject *should_defer;
    int defer;
    if (!PyArg_ParseTuple(args, "KO", &mi_addr, &should_defer)) {
        return NULL;
    }
    defer = PyObject_IsTrue(should_defer);
    if (defer == -1) {
        return NULL;
    }
    NRT_MemInfo_release((MemInfo*)mi_addr, defer);
    Py_RETURN_NONE;
}


static
void pyobject_dtor(void *ptr, void* info) {
    PyGILState_STATE gstate;
    PyObject *ownerobj = info;

    gstate = PyGILState_Ensure();   /* ensure the GIL */
    Py_DECREF(ownerobj);            /* release the python object */
    PyGILState_Release(gstate);     /* release the GIL */
}


/*
 * Create a new MemInfo with a owner PyObject
 */
static
PyObject*
meminfo_new(PyObject *self, PyObject *args) {
    PY_LONG_LONG addr_data;
    PyObject* ownerobj;
    MemInfo *mi;
    if (!PyArg_ParseTuple(args, "KO", &addr_data, &ownerobj)) {
        return NULL;
    }
    Py_INCREF(ownerobj);
    mi = NRT_MemInfo_new((void*)addr_data, pyobject_dtor, ownerobj);
    return Py_BuildValue("K", mi);
}

static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(memsys_set_atomic_inc_dec),
    declmethod(memsys_process_defer_dtor),
    declmethod(meminfo_new),
    declmethod(meminfo_acquire),
    declmethod(meminfo_release),
    { NULL },
#undef declmethod
};


MOD_INIT(_nrt_python) {
    PyObject *m;
    MOD_DEF(m, "_nrt_python", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;
    return MOD_SUCCESS_VAL(m);
}
