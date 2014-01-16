#include "_pymodule.h"
#include <structmember.h>
#include <string.h>

typedef struct DefnList
{
    PyObject *key;
    PyCFunctionWithKeywords func;
    struct DefnList *next;   /* link to next node or NULL */
} DefnList;


typedef struct DispatcherObject{
    PyObject_HEAD
    DefnList *defns;        /* singly-linked list */
} DispatcherObject;


static
void
Dispatcher_dealloc(DispatcherObject *self)
{
    DefnList *node, *tmpnode;;

    node = self->defns;
    while (node) {
        Py_DECREF(node->key);
        tmpnode = node;
        node = node->next;
        free(tmpnode);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static
int
Dispatcher_init(DispatcherObject *self, PyObject *args, PyObject *kwds)
{
    self->defns = NULL;
    return 0;
}


static
PyObject*
Dispatcher_Insert(DispatcherObject *self, PyObject *args)
{
    PyObject *key, *addr;
    DefnList *node;

    if (!PyArg_ParseTuple(args, "OO", &key, &addr)) {
        return NULL;
    }

    Py_INCREF(key);

    node = malloc(sizeof(DefnList));
    node->key = key;
    node->func = (PyCFunctionWithKeywords)PyLong_AsVoidPtr(addr);
    node->next = self->defns;

    self->defns = node;

    Py_RETURN_NONE;
}


static
PyObject*
Dispatcher_List(DispatcherObject *self, PyObject *args)
{
    DefnList *node;
    PyObject *retlist = PyList_New(0);
    PyObject *tup, *addr;;

    for (node = self->defns; node; node = node->next) {
        addr = PyLong_FromVoidPtr(node->func);
        tup = PyTuple_Pack(2, node->key, addr);
        Py_XDECREF(addr);
        if (!tup)
            return NULL;
        if (-1 == PyList_Append(retlist, tup))
            return NULL;
    }
    return retlist;
}


static
DefnList* Dispatcher_find(DispatcherObject* self, PyObject* match)
{
    /*
    TODO cache the most recent node by moving it to the head
    */
    DefnList *node;
    int cmpresult;
    for (node = self->defns; node; node = node->next) {
        cmpresult = PyObject_RichCompareBool(node->key, match, Py_EQ);
        if (cmpresult == 1) {
            break;
        } else if (cmpresult == -1) {
            return NULL;
        }
    }
    return node;
}


static
PyObject*
Dispatcher_Find(DispatcherObject *self, PyObject *args)
{
    DefnList *node;
    PyObject *match, *tup, *addr;
    if (!PyArg_ParseTuple(args, "O", &match)) {
        return NULL;
    }

    node = Dispatcher_find(self, match);

    if (NULL == node) {
        PyErr_SetString(PyExc_TypeError, "No matching definition");
        return NULL;
    }

    addr = PyLong_FromVoidPtr(node->func);
    tup = PyTuple_Pack(2, node->key, addr);
    Py_XDECREF(addr);
    return tup;
}


static
PyObject*
Dispatcher_call(DispatcherObject *self, PyObject *args, PyObject *kws)
{
    PyObject *match, *argvals, *retval;
    DefnList *node;

    match = PyTuple_GET_ITEM(args, 0);
    argvals = PyTuple_GET_ITEM(args, 1);

    node = Dispatcher_find(self, match);

    if (NULL == node) {
        PyErr_SetString(PyExc_TypeError, "No matching definition");
        return NULL;
    }

    retval = node->func(NULL, argvals, kws);
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
    { "list", (PyCFunction)Dispatcher_List, METH_NOARGS,
      "return a list of definitions"},
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
