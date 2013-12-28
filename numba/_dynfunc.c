#include <Python.h>

static unsigned long long TheCounter = 0;
static char ModNameBuf[128];

static
PyObject*
make_function(PyObject *self, PyObject *args)
{
    PyObject * module = NULL;
    Py_ssize_t fnaddr = 0;

    PyMethodDef dummy[] = {
        { "dyncallable", 0, METH_KEYWORDS, NULL },
        { NULL },
    };

    PyMethodDef *functable = NULL;

    if (!PyArg_ParseTuple(args, "n", &fnaddr)) {
        return NULL;
    }

    ((void**)dummy)[1] = (void*)fnaddr;

    /* FIXME Need to figure out a way to free the table */
    functable = malloc(sizeof(dummy));
    memcpy(functable, dummy, sizeof(dummy));

    sprintf(ModNameBuf, "dynmod%llu", TheCounter++);

    /* FIXME Python3 */
    module = Py_InitModule(ModNameBuf, functable);

    Py_XINCREF(module);
    return module;
}

static PyMethodDef core_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(make_function),
    { NULL },
#undef declmethod
};


#if (PY_MAJOR_VERSION >= 3)
    struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_dynfunc",
        NULL,
        -1,
        core_methods,
        NULL, NULL, NULL, NULL
    };
#define INITERROR return NULL
    PyObject *
    PyInit__dynfunc(void)
#else
#define INITERROR return
    PyMODINIT_FUNC
    init_dynfunc(void)
#endif
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *module = PyModule_Create( &module_def );
#else
        PyObject *module = Py_InitModule("_dynfunc", core_methods);
#endif
        if (module == NULL){
            INITERROR;
        }
#if PY_MAJOR_VERSION >= 3
        return module;
#endif
    }