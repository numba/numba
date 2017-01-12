#define NUMBA_EXPORT_FUNC(_rettype) static _rettype
#define NUMBA_EXPORT_DATA(_vartype) static _vartype

#include "_nrt_python.c"

static PyObject *
memsys_shutdown(PyObject *self, PyObject *args) {
    NRT_MemSys_shutdown();
    Py_RETURN_NONE;
}

static PyObject *
memsys_use_cpython_allocator(PyObject *self, PyObject *args) {
    NRT_MemSys_set_allocator(PyMem_RawMalloc,
                             PyMem_RawRealloc,
                             PyMem_RawFree);
    Py_RETURN_NONE;
}

static PyObject *
memsys_set_atomic_inc_dec(PyObject *self, PyObject *args) {
    PyObject *addr_inc_obj, *addr_dec_obj;
    void *addr_inc, *addr_dec;
    if (!PyArg_ParseTuple(args, "OO", &addr_inc_obj, &addr_dec_obj)) {
        return NULL;
    }
    addr_inc = PyLong_AsVoidPtr(addr_inc_obj);
    if(PyErr_Occurred()) return NULL;
    addr_dec = PyLong_AsVoidPtr(addr_dec_obj);
    if(PyErr_Occurred()) return NULL;
    NRT_MemSys_set_atomic_inc_dec(addr_inc, addr_dec);
    Py_RETURN_NONE;
}

static PyObject *
memsys_set_atomic_cas(PyObject *self, PyObject *args) {
    PyObject *addr_cas_obj;
    void *addr_cas;
    if (!PyArg_ParseTuple(args, "O", &addr_cas_obj)) {
        return NULL;
    }
    addr_cas = PyLong_AsVoidPtr(addr_cas_obj);
    if(PyErr_Occurred()) return NULL;
    NRT_MemSys_set_atomic_cas(addr_cas);
    Py_RETURN_NONE;
}

static PyObject *
memsys_get_stats_alloc(PyObject *self, PyObject *args) {
    return PyLong_FromSize_t(NRT_MemSys_get_stats_alloc());
}

static PyObject *
memsys_get_stats_free(PyObject *self, PyObject *args) {
    return PyLong_FromSize_t(NRT_MemSys_get_stats_free());
}

static PyObject *
memsys_get_stats_mi_alloc(PyObject *self, PyObject *args) {
    return PyLong_FromSize_t(NRT_MemSys_get_stats_mi_alloc());
}

static PyObject *
memsys_get_stats_mi_free(PyObject *self, PyObject *args) {
    return PyLong_FromSize_t(NRT_MemSys_get_stats_mi_free());
}


/*
 * Create a new MemInfo with a owner PyObject
 */
static PyObject *
meminfo_new(PyObject *self, PyObject *args) {
    PyObject *addr_data_obj;
    void *addr_data;
    PyObject *ownerobj;
    NRT_MemInfo *mi;
    if (!PyArg_ParseTuple(args, "OO", &addr_data_obj, &ownerobj)) {
        return NULL;
    }
    addr_data = PyLong_AsVoidPtr(addr_data_obj);
    if (PyErr_Occurred())
        return NULL;
    mi = meminfo_new_from_pyobject(addr_data, ownerobj);
    return PyLong_FromVoidPtr(mi);
}

/*
 * Create a new MemInfo with a new NRT allocation
 */
static PyObject *
meminfo_alloc(PyObject *self, PyObject *args) {
    NRT_MemInfo *mi;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "n", &size)) {
        return NULL;
    }
    mi = NRT_MemInfo_alloc(size);
    return PyLong_FromVoidPtr(mi);
}

/*
 * Like meminfo_alloc but set memory to zero after allocation and before
 * deallocation.
 */
static PyObject *
meminfo_alloc_safe(PyObject *self, PyObject *args) {
    NRT_MemInfo *mi;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "n", &size)) {
        return NULL;
    }
    mi = NRT_MemInfo_alloc_safe(size);
    return PyLong_FromVoidPtr(mi);
}

static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
#define declmethod_noargs(func) { #func , ( PyCFunction )func , METH_NOARGS, NULL }
    declmethod_noargs(memsys_use_cpython_allocator),
    declmethod_noargs(memsys_shutdown),
    declmethod(memsys_set_atomic_inc_dec),
    declmethod(memsys_set_atomic_cas),
    declmethod_noargs(memsys_get_stats_alloc),
    declmethod_noargs(memsys_get_stats_free),
    declmethod_noargs(memsys_get_stats_mi_alloc),
    declmethod_noargs(memsys_get_stats_mi_free),
    declmethod(meminfo_new),
    declmethod(meminfo_alloc),
    declmethod(meminfo_alloc_safe),
    { NULL },
#undef declmethod
};



static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr(value);           \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

#define declmethod(func) _declpointer(#func, &NRT_##func)

declmethod(adapt_ndarray_from_python);
declmethod(adapt_ndarray_to_python);
declmethod(adapt_buffer_from_python);
declmethod(MemInfo_alloc);
declmethod(MemInfo_alloc_safe);
declmethod(MemInfo_alloc_aligned);
declmethod(MemInfo_alloc_safe_aligned);
declmethod(MemInfo_alloc_dtor_safe);
declmethod(MemInfo_call_dtor);
declmethod(MemInfo_new_varsize);
declmethod(MemInfo_varsize_alloc);
declmethod(MemInfo_varsize_free);
declmethod(MemInfo_varsize_realloc);
declmethod(MemInfo_release);
declmethod(Allocate);
declmethod(Free);


#undef declmethod
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

MOD_INIT(_nrt_python) {
    PyObject *m;
    MOD_DEF(m, "_nrt_python", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    NRT_MemSys_init();
    if (init_nrt_python_module(m))
        return MOD_ERROR_VAL;

    Py_INCREF(&MemInfoType);
    PyModule_AddObject(m, "_MemInfo", (PyObject *) (&MemInfoType));

    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());

    return MOD_SUCCESS_VAL(m);
}
