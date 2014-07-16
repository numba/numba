#include "_pymodule.h"
#include <string.h>


/* A closure object is created for each call to make_function(), and stored
   as the resulting PyCFunction object's "self" pointer.  This allows
   proper lifetime management of some dependent data, and can in the
   future allow the raw function to know about its environment (e.g.
   the various enclosing lexical scopes).
   */

typedef struct {
    PyObject_HEAD
    /* The dynamically-filled method definition for the PyCFunction object
       using this closure. */
    PyMethodDef def;
    /* Other things could go in there, e.g. __globals__ and __closure__,
       or a reference to the LLVM function. */
} ClosureObject;


static int
closure_traverse(ClosureObject *clo, visitproc visit, void *arg)
{
    return 0;
}

static void
closure_dealloc(ClosureObject *clo)
{
    _PyObject_GC_UNTRACK((PyObject *) clo);
    PyObject_Free((void *) clo->def.ml_name);
    PyObject_Free((void *) clo->def.ml_doc);
    Py_TYPE(clo)->tp_free((PyObject *) clo);
}

static PyTypeObject ClosureType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_dynfunc._Closure",    /*tp_name*/
    sizeof(ClosureObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor) closure_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
    0,                         /* tp_doc */
    (traverseproc) closure_traverse, /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};


/* Return an owned piece of character data duplicating a Python string
   object's value. */
static char *
dup_string(PyObject *strobj)
{
    char *tmp, *str;
    tmp = PyString_AsString(strobj);
    if (tmp == NULL)
        return NULL;
    /* Using PyObject_Malloc allows this memory to be tracked for
       leaks. */
    str = PyObject_Malloc(strlen(tmp) + 1);
    if (str == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    strcpy(str, tmp);
    return str;
}

static ClosureObject *
closure_new(PyObject *module, PyObject *name, PyObject *doc, PyCFunction fnaddr)
{
    ClosureObject *clo = (ClosureObject *) PyType_GenericAlloc(&ClosureType, 0);
    if (clo == NULL)
        return NULL;

    clo->def.ml_name = dup_string(name);
    if (!clo->def.ml_name) {
        Py_DECREF(clo);
        return NULL;
    }
    clo->def.ml_meth = fnaddr;
    clo->def.ml_flags = METH_VARARGS | METH_KEYWORDS;
    clo->def.ml_doc = dup_string(doc);
    if (!clo->def.ml_doc) {
        Py_DECREF(clo);
        return NULL;
    }
    return clo;
}

/* Dynamically create a new C function object */
static
PyObject*
make_function(PyObject *self, PyObject *args)
{
    PyObject *module, *fname, *fdoc, *fnaddrobj;
    void *fnaddr;
    ClosureObject *closure;
    PyObject *funcobj;

    if (!PyArg_ParseTuple(args, "OOOO", &module, &fname, &fdoc, &fnaddrobj)) {
        return NULL;
    }

    fnaddr = PyLong_AsVoidPtr(fnaddrobj);
    if (fnaddr == NULL && PyErr_Occurred())
        return NULL;

    closure = closure_new(module, fname, fdoc, fnaddr);
    if (closure == NULL)
        return NULL;

    funcobj = PyCFunction_NewEx(&closure->def, (PyObject *) closure, module);
    Py_DECREF(closure);
    return funcobj;
}


static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(make_function),
    { NULL },
#undef declmethod
};


MOD_INIT(_dynfunc) {
    PyObject *m;
    MOD_DEF(m, "_dynfunc", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    if (PyType_Ready(&ClosureType))
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}
