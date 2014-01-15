#include "_pymodule.h"
#include <string.h>


#if (PY_MAJOR_VERSION >= 3)
    #define PyString_AsString PyUnicode_AsUTF8
#endif

static
PyObject*
make_function(PyObject *self, PyObject *args)
{
    PyObject *module, *fname, *fdoc;
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

    mlname = malloc(szname);
    mldoc = malloc(szdoc);
    strcpy(mlname, name);
    strcpy(mldoc, doc);

    /* FIXME Need to figure out a way to free the mallocs */
    desc = malloc(sizeof(PyMethodDef));
    desc->ml_name = mlname;
    desc->ml_meth = (void*)fnaddr;
    desc->ml_flags = METH_VARARGS | METH_KEYWORDS;
    desc->ml_doc = mldoc;

    return PyCFunction_NewEx(desc, NULL, module);
}


static
PyObject*
set_arbitrary_addr(PyObject* self, PyObject* args){
    PyObject *obj;
    union {
        Py_ssize_t intval;
        void **ptr;
    } target;
    if (!PyArg_ParseTuple(args, "nO", &target.intval, &obj)) {
        return NULL;
    }
    *target.ptr = (void*)obj;
    Py_RETURN_NONE;
}


static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(make_function),
    declmethod(set_arbitrary_addr),
    { NULL },
#undef declmethod
};


MOD_INIT(_dynfunc) {
    PyObject *m;
    MOD_DEF(m, "_dynfunc", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}
