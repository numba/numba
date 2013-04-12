#include <Python.h>

static PyObject*
memoryview_get_buffer(PyObject *self, PyObject *args){
    PyObject *mv;
    if (!PyArg_ParseTuple(args, "O", &mv))
        return 0;

    if (!PyMemoryView_Check(mv))
        return 0;

    Py_buffer* buf = PyMemoryView_GET_BUFFER(mv);
    return PyLong_FromVoidPtr(buf->buf);
}


static PyMethodDef core_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(memoryview_get_buffer),
    { NULL },
#undef declmethod
};


// Module main function, hairy because of py3k port

#if (PY_MAJOR_VERSION >= 3)
    struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "mviewbuf",
        NULL,
        -1,
        core_methods,
        NULL, NULL, NULL, NULL
    };
#define INITERROR return NULL
    PyObject *
    PyInit_mviewbuf(void)
#else
#define INITERROR return
    PyMODINIT_FUNC
    initmviewbuf(void)
#endif
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *module = PyModule_Create( &module_def );
#else
        PyObject *module = Py_InitModule("mviewbuf", core_methods);
#endif
        if (module == NULL)
            INITERROR;
#if PY_MAJOR_VERSION >= 3
        
        return module;
#endif
    }

