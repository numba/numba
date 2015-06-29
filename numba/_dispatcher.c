#include "_pymodule.h"

#include <string.h>
#include <time.h>
#include <assert.h>

#include "_dispatcher.h"
#include "_typeof.h"


typedef struct DispatcherObject{
    PyObject_HEAD
    /* Holds borrowed references to PyCFunction objects */
    dispatcher_t *dispatcher;
    char can_compile;        /* Can auto compile */
    /* Borrowed references */
    PyObject *firstdef, *fallbackdef, *interpdef;
    /* Whether to fold named arguments and default values (false for lifted loops)*/
    int fold_args;
    /* Whether the last positional argument is a stararg */
    int has_stararg;
    /* Tuple of argument names */
    PyObject *argnames;
    /* Tuple of default values */
    PyObject *defargs;
} DispatcherObject;


static int
Dispatcher_traverse(DispatcherObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->defargs);
    return 0;
}

static void
Dispatcher_dealloc(DispatcherObject *self)
{
    Py_XDECREF(self->argnames);
    Py_XDECREF(self->defargs);
    dispatcher_del(self->dispatcher);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


static int
Dispatcher_init(DispatcherObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmaddrobj;
    void *tmaddr;
    int argct;
    int has_stararg = 0;

    if (!PyArg_ParseTuple(args, "OiiO!O!|i", &tmaddrobj, &argct,
                          &self->fold_args,
                          &PyTuple_Type, &self->argnames,
                          &PyTuple_Type, &self->defargs,
                          &has_stararg)) {
        return -1;
    }
    Py_INCREF(self->argnames);
    Py_INCREF(self->defargs);
    tmaddr = PyLong_AsVoidPtr(tmaddrobj);
    self->dispatcher = dispatcher_new(tmaddr, argct);
    self->can_compile = 1;
    self->firstdef = NULL;
    self->fallbackdef = NULL;
    self->interpdef = NULL;
    self->has_stararg = has_stararg;
    return 0;
}

static PyObject *
Dispatcher_clear(DispatcherObject *self, PyObject *args)
{
    dispatcher_clear(self->dispatcher);
    Py_RETURN_NONE;
}

static
PyObject*
Dispatcher_Insert(DispatcherObject *self, PyObject *args)
{
    PyObject *sigtup, *cfunc;
    int i, sigsz;
    int *sig;
    int objectmode = 0;
    int interpmode = 0;

    if (!PyArg_ParseTuple(args, "OO|ii", &sigtup,
                          &cfunc, &objectmode, &interpmode)) {
        return NULL;
    }

    if (!interpmode && !PyObject_TypeCheck(cfunc, &PyCFunction_Type) ) {
        PyErr_SetString(PyExc_TypeError, "must be builtin_function_or_method");
        return NULL;
    }

    sigsz = PySequence_Fast_GET_SIZE(sigtup);
    sig = malloc(sigsz * sizeof(int));

    for (i = 0; i < sigsz; ++i) {
        sig[i] = PyLong_AsLong(PySequence_Fast_GET_ITEM(sigtup, i));
    }

    if (!interpmode) {
        /* The reference to cfunc is borrowed; this only works because the
           derived Python class also stores an (owned) reference to cfunc. */
        dispatcher_add_defn(self->dispatcher, sig, (void*) cfunc);

        /* Add first definition */
        if (!self->firstdef) {
            self->firstdef = cfunc;
        }
    }
    /* Add pure python fallback */
    if (!self->fallbackdef && objectmode){
        self->fallbackdef = cfunc;
    }
    /* Add interpeter fallback */
    if (!self->interpdef && interpmode) {
        self->interpdef = cfunc;
    }

    free(sig);

    Py_RETURN_NONE;
}


static
void explain_issue(PyObject *dispatcher, PyObject *args, PyObject *kws,
                   const char *method_name, const char *default_msg)
{
    PyObject *callback, *result;
    callback = PyObject_GetAttrString(dispatcher, method_name);
    if (!callback) {
        PyErr_SetString(PyExc_TypeError, default_msg);
        return;
    }
    result = PyObject_Call(callback, args, kws);
    Py_DECREF(callback);
    if (result != NULL) {
        PyErr_Format(PyExc_RuntimeError, "%s must raise an exception",
                     method_name);
        Py_DECREF(result);
    }
}

static
void explain_ambiguous(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    explain_issue(dispatcher, args, kws, "_explain_ambiguous",
                  "Ambigous overloading");
}

static
void explain_matching_error(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    explain_issue(dispatcher, args, kws, "_explain_matching_error",
                  "No matching definition");
}

static
int search_new_conversions(PyObject *dispatcher, PyObject *args, PyObject *kws)
{
    PyObject *callback, *result;
    int res;

    callback = PyObject_GetAttrString(dispatcher,
                                      "_search_new_conversions");
    if (!callback) {
        return -1;
    }
    result = PyObject_Call(callback, args, kws);
    Py_DECREF(callback);
    if (result == NULL) {
        return -1;
    }
    if (!PyBool_Check(result)) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError,
                        "_search_new_conversions() should return a boolean");
        return -1;
    }
    res = (result == Py_True) ? 1 : 0;
    Py_DECREF(result);
    return res;
}

/* A custom, fast, inlinable version of PyCFunction_Call() */
static PyObject *
call_cfunc(PyObject *cfunc, PyObject *args, PyObject *kws)
{
    PyCFunctionWithKeywords fn;
    assert(PyCFunction_Check(cfunc));
    assert(PyCFunction_GET_FLAGS(cfunc) == METH_VARARGS | METH_KEYWORDS);
    fn = (PyCFunctionWithKeywords) PyCFunction_GET_FUNCTION(cfunc);
    return fn(PyCFunction_GET_SELF(cfunc), args, kws);
}

static
PyObject*
compile_and_invoke(DispatcherObject *self, PyObject *args, PyObject *kws)
{
    /* Compile a new one */
    PyObject *cfa, *cfunc, *retval;
    cfa = PyObject_GetAttrString((PyObject*)self, "_compile_for_args");
    if (cfa == NULL)
        return NULL;

    /* NOTE: we call the compiled function ourselves instead of
       letting the Python derived class do it.  This is for proper
       behaviour of globals() in jitted functions (issue #476). */
    cfunc = PyObject_Call(cfa, args, kws);
    Py_DECREF(cfa);

    if (cfunc == NULL)
        return NULL;

    if (PyObject_TypeCheck(cfunc, &PyCFunction_Type)) {
        retval = call_cfunc(cfunc, args, kws);
    } else {
        // Re-enter interpreter
        retval = PyObject_Call(cfunc, args, kws);
    }
    Py_DECREF(cfunc);

    return retval;
}

static int
find_named_args(DispatcherObject *self, PyObject **pargs, PyObject **pkws)
{
    PyObject *oldargs = *pargs, *newargs;
    PyObject *kws = *pkws;
    Py_ssize_t pos_args = PyTuple_GET_SIZE(oldargs);
    Py_ssize_t named_args, total_args, i;
    Py_ssize_t func_args = PyTuple_GET_SIZE(self->argnames);
    Py_ssize_t defaults = PyTuple_GET_SIZE(self->defargs);
    /* Last parameter with a default value */
    Py_ssize_t last_def = (self->has_stararg)
                          ? func_args - 2
                          : func_args - 1;
    /* First parameter with a default value */
    Py_ssize_t first_def = last_def - defaults + 1;
    /* Minimum number of required arguments */
    Py_ssize_t minargs = first_def;

    if (kws != NULL)
        named_args = PyDict_Size(kws);
    else
        named_args = 0;
    total_args = pos_args + named_args;
    if (!self->has_stararg && total_args > func_args) {
        PyErr_Format(PyExc_TypeError,
                     "too many arguments: expected %d, got %d",
                     (int) func_args, (int) total_args);
        return -1;
    }
    else if (total_args < minargs) {
        if (minargs == func_args)
            PyErr_Format(PyExc_TypeError,
                         "not enough arguments: expected %d, got %d",
                         (int) minargs, (int) total_args);
        else
            PyErr_Format(PyExc_TypeError,
                         "not enough arguments: expected at least %d, got %d",
                         (int) minargs, (int) total_args);
        return -1;
    }
    newargs = PyTuple_New(func_args);
    if (!newargs)
        return -1;
    /* First pack the stararg */
    if (self->has_stararg) {
        Py_ssize_t stararg_size = Py_MAX(0, pos_args - func_args + 1);
        PyObject *stararg = PyTuple_New(stararg_size);
        if (!stararg) {
            Py_DECREF(newargs);
            return -1;
        }
        for (i = 0; i < stararg_size; i++) {
            PyObject *value = PyTuple_GET_ITEM(oldargs, func_args - 1 + i);
            Py_INCREF(value);
            PyTuple_SET_ITEM(stararg, i, value);
        }
        /* Put it in last position */
        PyTuple_SET_ITEM(newargs, func_args - 1, stararg);

    }
    for (i = 0; i < pos_args; i++) {
        PyObject *value = PyTuple_GET_ITEM(oldargs, i);
        if (self->has_stararg && i >= func_args - 1) {
            /* Skip stararg */
            break;
        }
        Py_INCREF(value);
        PyTuple_SET_ITEM(newargs, i, value);
    }

    /* Iterate over missing positional arguments, try to find them in
       named arguments or default values. */
    for (i = pos_args; i < func_args; i++) {
        PyObject *name = PyTuple_GET_ITEM(self->argnames, i);
        if (self->has_stararg && i >= func_args - 1) {
            /* Skip stararg */
            break;
        }
        if (kws != NULL) {
            /* Named argument? */
            PyObject *value = PyDict_GetItem(kws, name);
            if (value != NULL) {
                Py_INCREF(value);
                PyTuple_SET_ITEM(newargs, i, value);
                named_args--;
                continue;
            }
        }
        if (i >= first_def && i <= last_def) {
            /* Argument has a default value? */
            PyObject *value = PyTuple_GET_ITEM(self->defargs, i - first_def);
            Py_INCREF(value);
            PyTuple_SET_ITEM(newargs, i, value);
            continue;
        }
        else if (i < func_args - 1 || !self->has_stararg) {
            PyErr_Format(PyExc_TypeError,
                         "missing argument '%s'",
                         PyString_AsString(name));
            Py_DECREF(newargs);
            return -1;
        }
    }
    if (named_args) {
        PyErr_Format(PyExc_TypeError,
                     "some keyword arguments unexpected");
        Py_DECREF(newargs);
        return -1;
    }
    *pargs = newargs;
    *pkws = NULL;
    return 0;
}

static PyObject*
Dispatcher_call(DispatcherObject *self, PyObject *args, PyObject *kws)
{
    PyObject *tmptype, *retval = NULL;
    int *tys;
    int argct;
    int i;
    int prealloc[24];
    int matches;
    PyObject *cfunc;

    if (self->fold_args) {
        if (find_named_args(self, &args, &kws))
            return NULL;
    }
    else
        Py_INCREF(args);
    /* Now we own a reference to args */

    argct = PySequence_Fast_GET_SIZE(args);

    if (argct < sizeof(prealloc) / sizeof(int))
        tys = prealloc;
    else
        tys = malloc(argct * sizeof(int));

    for (i = 0; i < argct; ++i) {
        tmptype = PySequence_Fast_GET_ITEM(args, i);
        tys[i] = typeof_typecode((PyObject *) self, tmptype);
        if (tys[i] == -1)
            goto CLEANUP;
    }

    /* We only allow unsafe conversions if compilation of new specializations
       has been disabled. */
    cfunc = dispatcher_resolve(self->dispatcher, tys, &matches,
                               !self->can_compile);

    if (matches == 0 && !self->can_compile) {
        /*
         * If we can't compile a new specialization, look for
         * matching signatures for which conversions haven't been
         * registered on the C++ TypeManager.
         */
        int res = search_new_conversions((PyObject *) self, args, kws);
        if (res < 0) {
            retval = NULL;
            goto CLEANUP;
        }
        if (res > 0) {
            /* Retry with the newly registered conversions */
            cfunc = dispatcher_resolve(self->dispatcher, tys, &matches,
                                       !self->can_compile);
        }
    }

    if (matches == 1) {
        /* Definition is found */
        retval = call_cfunc(cfunc, args, kws);
    } else if (matches == 0) {
        /* No matching definition */
        if (self->can_compile) {
            retval = compile_and_invoke(self, args, kws);
        } else if (self->fallbackdef) {
            /* Have object fallback */
            retval = call_cfunc(self->fallbackdef, args, kws);
        } else {
            /* Raise TypeError */
            explain_matching_error((PyObject *) self, args, kws);
            retval = NULL;
        }
    } else if (self->can_compile) {
        /* Ambiguous, but are allowed to compile */
        retval = compile_and_invoke(self, args, kws);
    } else {
        /* Ambiguous */
        explain_ambiguous((PyObject *) self, args, kws);
        retval = NULL;
    }

CLEANUP:
    if (tys != prealloc)
        free(tys);
    Py_DECREF(args);

    return retval;
}

static PyMethodDef Dispatcher_methods[] = {
    { "_clear", (PyCFunction)Dispatcher_clear, METH_NOARGS, NULL },
    { "_insert", (PyCFunction)Dispatcher_Insert, METH_VARARGS,
      "insert new definition"},
    { NULL },
};

static PyMemberDef Dispatcher_members[] = {
    {"_can_compile", T_BOOL, offsetof(DispatcherObject, can_compile), 0},
    {NULL}  /* Sentinel */
};


static PyTypeObject DispatcherType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                                           /* ob_size */
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_dispatcher.Dispatcher",                    /* tp_name */
    sizeof(DispatcherObject),                    /* tp_basicsize */
    0,                                           /* tp_itemsize */
    (destructor)Dispatcher_dealloc,              /* tp_dealloc */
    0,                                           /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_compare */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    0,                                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    (PyCFunctionWithKeywords)Dispatcher_call,    /* tp_call*/
    0,                                           /* tp_str*/
    0,                                           /* tp_getattro*/
    0,                                           /* tp_setattro*/
    0,                                           /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags*/
    "Dispatcher object",                         /* tp_doc */
    (traverseproc) Dispatcher_traverse,          /* tp_traverse */
    0,                                           /* tp_clear */
    0,                                           /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    0,                                           /* tp_iternext */
    Dispatcher_methods,                          /* tp_methods */
    Dispatcher_members,                          /* tp_members */
    0,                                           /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    (initproc)Dispatcher_init,                   /* tp_init */
    0,                                           /* tp_alloc */
    0,                                           /* tp_new */
};


static PyObject *compute_fingerprint(PyObject *self, PyObject *args)
{
    PyObject *val;
    if (!PyArg_ParseTuple(args, "O:compute_fingerprint", &val))
        return NULL;
    return typeof_compute_fingerprint(val);
}

static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
    declmethod(typeof_init),
    declmethod(compute_fingerprint),
    { NULL },
#undef declmethod
};


MOD_INIT(_dispatcher) {
    PyObject *m;
    MOD_DEF(m, "_dispatcher", "No docs", ext_methods)
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
