#include <Python.h>
#include <string.h>


#if (PY_MAJOR_VERSION >= 3)
    #define PyString_AsString PyUnicode_AsUTF8
#endif

static
PyObject*
make_function(PyObject *self, PyObject *args)
{
    PyObject *module, *fname, *fdoc, *dict;
    Py_ssize_t fnaddr;
    PyMethodDef *desc;
    char *doc, *name;
    char *mlname, *mldoc;
    size_t szdoc, szname;

    if (!PyArg_ParseTuple(args, "OOOn", &module, &fname, &fdoc, &fnaddr)) {
        return NULL;
    }

    doc = PyString_AsString(fdoc);
    name = PyString_AsString(fname);
    szdoc = strlen(doc) + 1;
    szname = strlen(name) + 1;
    dict = PyObject_GetAttrString(module, "__dict__");

    mlname = malloc(szname);
    mldoc = malloc(szdoc);
    strcpy(mlname, name);
    strcpy(mldoc, doc);

    /* FIXME Need to figure out a way to free the mallocs */
    desc = malloc(sizeof(PyMethodDef));
    desc->ml_name = mlname;
    desc->ml_meth = (void*)fnaddr;
    desc->ml_flags = METH_KEYWORDS;
    desc->ml_doc = mldoc;

    return PyCFunction_NewEx(desc, dict, module);
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