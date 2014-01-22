#include "_pymodule.h"
#include <structmember.h>
#include <string.h>
#include <time.h>
#include "_dispatcher.h"


typedef struct DispatcherObject{
    PyObject_HEAD
    void *dispatcher;
} DispatcherObject;


static
void
Dispatcher_dealloc(DispatcherObject *self)
{
    dispatcher_del(self->dispatcher);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static
int
Dispatcher_init(DispatcherObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmaddrobj;
    void *tmaddr;
    int argct;
    if (!PyArg_ParseTuple(args, "Oi", &tmaddrobj, &argct)) {

    }
    tmaddr = PyLong_AsVoidPtr(tmaddrobj);
    self->dispatcher = dispatcher_new(tmaddr, argct);
    return 0;
}


static
PyObject*
Dispatcher_Insert(DispatcherObject *self, PyObject *args)
{
    PyObject *sigtup, *addrobj;
    void *addr;
    int i, sigsz;
    int *sig;

    if (!PyArg_ParseTuple(args, "OO", &sigtup, &addrobj)) {
        return NULL;
    }
    addr = PyLong_AsVoidPtr(addrobj);
    sigsz = PySequence_Fast_GET_SIZE(sigtup);
    sig = malloc(sigsz * sizeof(int));

    for (i = 0; i < sigsz; ++i) {
        sig[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(sigtup, i));
    }

    dispatcher_add_defn(self->dispatcher, sig, (void*)addr);

    free(sig);

    Py_RETURN_NONE;
}


static
PyObject*
Dispatcher_Find(DispatcherObject *self, PyObject *args)
{
    PyObject *sigtup;
    int i, sigsz;
    int *sig;
    void *out;

    if (!PyArg_ParseTuple(args, "O", &sigtup)) {
        return NULL;
    }

    sigsz = PySequence_Fast_GET_SIZE(sigtup);

    sig = malloc(sigsz * sizeof(int));
    for (i = 0; i < sigsz; ++i) {
        sig[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(sigtup, i));
    }

    out = dispatcher_resolve(self->dispatcher, sig);

    free(sig);

    return PyLong_FromVoidPtr(out);
}

static
int typecode_fallback(void *dispatcher, PyObject *val) {
    PyObject *dpmod, *typeof_pyval, *tmptype, *tmpstr;
    int typecode;

    // Go back to the interpreter

    dpmod = PyImport_ImportModule("numba.dispatcher");
    typeof_pyval = PyObject_GetAttrString(dpmod, "typeof_pyval");
    tmptype = PyObject_CallFunctionObjArgs(typeof_pyval, val, NULL);
    tmpstr = PyObject_Str(tmptype);
    typecode = dispatcher_get_type(dispatcher, PyString_AsString(tmpstr));

    Py_XDECREF(tmptype);
    Py_XDECREF(tmpstr);
    Py_XDECREF(typeof_pyval);
    Py_XDECREF(dpmod);

    return typecode;
}

static
int typecode(void *dispatcher, PyObject *val) {
    if (PyObject_TypeCheck(val, &PyInt_Type)
        || PyObject_TypeCheck(val, &PyLong_Type)) {
        switch (sizeof(void*)) {
        case 4: /* 32-bit */
            return dispatcher_get_type(dispatcher, "int32");
        case 8: /* 64-bit */
            return dispatcher_get_type(dispatcher, "int64");
        }
    } else if (PyObject_TypeCheck(val, &PyFloat_Type)) {
        return dispatcher_get_type(dispatcher, "float64");
    } else if (PyObject_TypeCheck(val, &PyComplex_Type)) {
        return dispatcher_get_type(dispatcher, "complex128");
    }
    /*
    Add array handling
    */

    return typecode_fallback(dispatcher, val);
}


static
PyObject*
Dispatcher_call(DispatcherObject *self, PyObject *args, PyObject *kws)
{
    PyObject *tmptype, *retval;
    int *tys;
    int argct;
    int i;
    int prealloc[24];
    PyCFunctionWithKeywords fn;

    argct = PySequence_Fast_GET_SIZE(args);

    if (argct < sizeof(prealloc) / sizeof(int))
        tys = prealloc;
    else
        tys = malloc(argct * sizeof(int));

    for (i = 0; i < argct; ++i) {
        tmptype = PySequence_Fast_GET_ITEM(args, i);
        tys[i] = typecode(self->dispatcher, tmptype);
    }

    fn = (PyCFunctionWithKeywords)dispatcher_resolve(self->dispatcher, tys);
    if (!fn) {
        PyErr_SetString(PyExc_TypeError, "No matching definition");
        return NULL;
    }

    retval = fn(NULL, args, kws);

    if (tys != prealloc)
        free(tys);

    return retval;

}


static PyMemberDef Dispatcher_members[] = {
    {NULL},
};


static PyMethodDef Dispatcher_methods[] = {
    { "insert", (PyCFunction)Dispatcher_Insert, METH_VARARGS,
      "insert new definition"},
    { "find", (PyCFunction)Dispatcher_Find, METH_VARARGS,
      "find matching definition and return a tuple of (argtypes, callable)"},
    { NULL },
};



static PyTypeObject DispatcherType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_dispatcher.Dispatcher",        /*tp_name*/
    sizeof(DispatcherObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Dispatcher_dealloc,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    (PyCFunctionWithKeywords)Dispatcher_call,           /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    "Dispatcher object",       /* tp_doc */
    0,		                   /* tp_traverse */
    0,		                   /* tp_clear */
    0,                         /*tp_richcompare */
    0,		                   /* tp_weaklistoffset */
    0,		                   /* tp_iter */
    0,		                   /* tp_iternext */
    Dispatcher_methods,        /* tp_methods */
    Dispatcher_members,        /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Dispatcher_init, /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};



MOD_INIT(_dispatcher) {
    PyObject *m;
    MOD_DEF(m, "_dispatcher", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    DispatcherType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&DispatcherType) < 0) {
        return MOD_ERROR_VAL;
    }
    Py_INCREF(&DispatcherType);
    PyModule_AddObject(m, "Dispatcher", (PyObject*)(&DispatcherType));

    return MOD_SUCCESS_VAL(m);
}
