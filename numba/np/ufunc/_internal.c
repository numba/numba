#include "_internal.h"
#include "Python.h"

/* A small object that handles deallocation of some of a PyUFunc's fields */
typedef struct {
    PyObject_HEAD
    /* Borrowed reference */
    PyUFuncObject *ufunc;
    /* Owned reference to ancillary object */
    PyObject *object;
} PyUFuncCleaner;

PyTypeObject PyUFuncCleaner_Type;


static PyObject *
cleaner_new(PyUFuncObject *ufunc, PyObject *object)
{
    PyUFuncCleaner *obj = PyObject_New(PyUFuncCleaner, &PyUFuncCleaner_Type);
    if (obj != NULL) {
        obj->ufunc = ufunc;
        Py_XINCREF(object);
        obj->object = object;
    }
    return (PyObject *) obj;
}

/* Deallocate the PyArray_malloc calls */
static void
cleaner_dealloc(PyUFuncCleaner *self)
{
    PyUFuncObject *ufunc = self->ufunc;
    Py_XDECREF(self->object);
    if (ufunc->functions)
        PyArray_free(ufunc->functions);
    if (ufunc->types)
        PyArray_free((void *)ufunc->types);
    if (ufunc->data)
        PyArray_free((void *)ufunc->data);
    PyObject_Del(self);
}

PyTypeObject PyUFuncCleaner_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "numba._UFuncCleaner",                      /* tp_name*/
    sizeof(PyUFuncCleaner),                     /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    (destructor) cleaner_dealloc,               /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
    0,                                          /* tp_finalize */
    0,                                          /* tp_vectorcall */
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 12)
/* This was introduced first in 3.12
 * https://github.com/python/cpython/issues/91051
 */
    0,                                           /* tp_watched */
#endif
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 13)
/* This was introduced in 3.13
 * https://github.com/python/cpython/pull/114900
 */
    0,                                           /* tp_versions_used */
#endif

/* WARNING: Do not remove this, only modify it! It is a version guard to
 * act as a reminder to update this struct on Python version update! */
#if (PY_MAJOR_VERSION == 3)
#if ! (NB_SUPPORTED_PYTHON_MINOR)
#error "Python minor version is not supported."
#endif
#else
#error "Python major version is not supported."
#endif
/* END WARNING*/
};

/* ______________________________________________________________________
 * DUFunc: A call-time (hence dynamic) specializable ufunc.
 */

typedef struct {
    PyObject_HEAD
    PyObject      * dispatcher;
    PyUFuncObject * ufunc;
    PyObject      * keepalive;
    int             frozen;
} PyDUFuncObject;

static void
dufunc_dealloc(PyDUFuncObject *self)
{
    /* Note: There is no need to call PyArray_free() on
       self->ufunc->ptr, since ufunc_dealloc() will do it for us. */
    Py_XDECREF(self->ufunc);
    Py_XDECREF(self->dispatcher);
    Py_XDECREF(self->keepalive);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
dufunc_repr(PyDUFuncObject *dufunc)
{
    return PyString_FromFormat("<numba._DUFunc '%s'>", dufunc->ufunc->name);
}

static PyObject *
dufunc_call(PyDUFuncObject *self, PyObject *args, PyObject *kws)
{
    PyObject *result=NULL, *method=NULL;

    result = PyUFunc_Type.tp_call((PyObject *)self->ufunc, args, kws);
    if ((!self->frozen) &&
            (result == NULL) &&
            (PyErr_Occurred()) &&
            (PyErr_ExceptionMatches(PyExc_TypeError))) {

        /* Break back into Python when we fail at dispatch. */
        PyErr_Clear();
        method = PyObject_GetAttrString((PyObject*)self, "_compile_for_args");

        if (method) {
            result = PyObject_Call(method, args, kws);
            if (result) {
                Py_DECREF(result);
                result = PyUFunc_Type.tp_call((PyObject *)self->ufunc, args,
                                              kws);
            }
        }
        Py_XDECREF(method);
    }
    return result;
}

static Py_ssize_t
_get_nin(PyObject * py_func_obj)
{
    int result = -1;
    PyObject *inspect=NULL, *getargspec=NULL, *argspec=NULL, *args=NULL;

    inspect = PyImport_ImportModule("inspect");
    if (!inspect) goto _get_nin_cleanup;
    getargspec = PyObject_GetAttrString(inspect, "getfullargspec");
    if (!getargspec) goto _get_nin_cleanup;
    argspec = PyObject_CallFunctionObjArgs(getargspec, py_func_obj, NULL);
    if (!argspec) goto _get_nin_cleanup;
    args = PyObject_GetAttrString(argspec, "args");
    if (!args) goto _get_nin_cleanup;
    result = PyList_Size(args);

  _get_nin_cleanup:
    Py_XDECREF(args);
    Py_XDECREF(argspec);
    Py_XDECREF(getargspec);
    Py_XDECREF(inspect);
    return result;
}

static int
dufunc_init(PyDUFuncObject *self, PyObject *args, PyObject *kws)
{
    PyObject *dispatcher=NULL, *keepalive=NULL, *py_func_obj=NULL, *tmp;
    PyUFuncObject *ufunc=NULL;
    int identity=PyUFunc_None;
    int nin=-1, nout=1, idx;
    const char *name=NULL, *doc=NULL;

    static char * kwlist[] = {"dispatcher", "identity", "_keepalive", "nin",
                              "nout", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kws, "O|iO!nn", kwlist,
                                     &dispatcher, &identity,
                                     &PyList_Type, &keepalive, &nin, &nout)) {
        return -1;
    }

    py_func_obj = PyObject_GetAttrString(dispatcher, "py_func");
    if (!py_func_obj) {
        return -1;
    }

    if (nin < 0) {
        nin = (int)_get_nin(py_func_obj);
        if ((nin < 0) || (PyErr_Occurred())) {
            Py_XDECREF(py_func_obj);
            return -1;
        }
    }

    /* Construct the UFunc. */
    tmp = PyObject_GetAttrString(py_func_obj, "__name__");
    if (tmp) {
        name = PyString_AsString(tmp);
    }
    Py_XDECREF(tmp);
    tmp = PyObject_GetAttrString(py_func_obj, "__doc__");
    if (tmp && (tmp != Py_None)) {
        doc = PyString_AsString(tmp);
    }
    Py_XDECREF(tmp);
    tmp = NULL;
    Py_XDECREF(py_func_obj);
    py_func_obj = NULL;
    if (!name) {
        return -1;
    }
    ufunc = (PyUFuncObject *)PyUFunc_FromFuncAndData(NULL, NULL, NULL, 0,
                                                     nin, nout, identity,
                                                     name, doc, 0);
    if (!ufunc) {
        return -1;
    }

    for (idx = 0; idx < nout; idx++) {
        ufunc->op_flags[idx + nin] |= NPY_ITER_READWRITE | NPY_ITER_UPDATEIFCOPY | NPY_ITER_ALLOCATE;
    }

    /* Construct a keepalive list if none was given. */
    if (!keepalive) {
        keepalive = PyList_New(0);
        if (!keepalive) {
            Py_XDECREF(ufunc);
            return -1;
        }
    } else {
        Py_INCREF(keepalive);
    }

    tmp = self->dispatcher;
    Py_INCREF(dispatcher);
    self->dispatcher = dispatcher;
    Py_XDECREF(tmp);

    tmp = (PyObject*)self->ufunc;
    self->ufunc = ufunc;
    Py_XDECREF(tmp);

    tmp = self->keepalive;
    /* Already incref'ed, either by PyList_New(), or else clause, both above. */
    self->keepalive = keepalive;
    Py_XDECREF(tmp);

    self->frozen = 0;

    return 0;
}

static PyMemberDef dufunc_members[] = {
    {"_dispatcher", T_OBJECT_EX, offsetof(PyDUFuncObject, dispatcher), 0,
         "Dispatcher object for the core Python function."},
    {"ufunc", T_OBJECT_EX, offsetof(PyDUFuncObject, ufunc), 0,
         "Numpy Ufunc for the dynamic ufunc."},
    {"_keepalive", T_OBJECT_EX, offsetof(PyDUFuncObject, keepalive), 0,
         "List of objects to keep alive during life of dufunc."},
    {NULL}
};

/* ____________________________________________________________
 * Shims to expose ufunc methods.
 */

static struct _ufunc_dispatch {
    /* Note that the following may also hold `_PyCFunctionFastWithKeywords` */
    PyCFunctionWithKeywords ufunc_reduce;
    PyCFunctionWithKeywords ufunc_accumulate;
    PyCFunctionWithKeywords ufunc_reduceat;
    PyCFunctionWithKeywords ufunc_outer;
    PyCFunction ufunc_at;
} ufunc_dispatch;

static int
init_ufunc_dispatch(int *numpy_uses_fastcall)
{
    int result = 0;
    PyMethodDef * crnt = PyUFunc_Type.tp_methods;
    const char * crnt_name = NULL;
    for (; crnt->ml_name != NULL; crnt++) {
        crnt_name = crnt->ml_name;
        switch(crnt_name[0]) {
        case 'a':
            if (strncmp(crnt_name, "accumulate", 11) == 0) {
                ufunc_dispatch.ufunc_accumulate =
                    (PyCFunctionWithKeywords)crnt->ml_meth;
            } else if (strncmp(crnt_name, "at", 3) == 0) {
                ufunc_dispatch.ufunc_at = crnt->ml_meth;
            } else {
                result = -1;
            }
            break;
        case 'o':
            if (strncmp(crnt_name, "outer", 6) == 0) {
                ufunc_dispatch.ufunc_outer =
                    (PyCFunctionWithKeywords)crnt->ml_meth;
            } else {
                result = -1;
            }
            break;
        case 'r':
            if (strncmp(crnt_name, "reduce", 7) == 0) {
                ufunc_dispatch.ufunc_reduce =
                    (PyCFunctionWithKeywords)crnt->ml_meth;
            } else if (strncmp(crnt_name, "reduceat", 9) == 0) {
                ufunc_dispatch.ufunc_reduceat =
                    (PyCFunctionWithKeywords)crnt->ml_meth;
            } else if (strncmp(crnt_name, "resolve_dtypes", 15) == 0) {
              /* Ignored */
            } else {
                result = -1;
            }
            break;
        case '_':
            // We ignore private methods
            break;
        default:
            result = -1; /* Unknown method */
        }
        if (result < 0) break;

        /* Check whether NumPy uses fastcall (ufunc.at never uses it) */
        if (strncmp(crnt_name, "at", 3) != 0) {
            if (*numpy_uses_fastcall == -1) {
                *numpy_uses_fastcall = crnt->ml_flags & METH_FASTCALL;
            }
            else if (*numpy_uses_fastcall != (crnt->ml_flags & METH_FASTCALL)) {
                PyErr_Format(PyExc_RuntimeError,
                    "ufunc.%s() flags do not match numpy_uses_fastcall",
                    crnt_name);
                return -1;
            }
        }
    }
    if (result == 0) {
        /* Sanity check. */
        result = ((ufunc_dispatch.ufunc_reduce != NULL)
                  && (ufunc_dispatch.ufunc_accumulate != NULL)
                  && (ufunc_dispatch.ufunc_reduceat != NULL)
                  && (ufunc_dispatch.ufunc_outer != NULL)
                  && (ufunc_dispatch.ufunc_at != NULL)
                  );
    } else {
        char const * const fmt = "Unexpected ufunc method %s()";
        PyErr_Format(PyExc_RuntimeError, fmt, crnt_name);
    }

    return result;
}


static PyObject *
dufunc_reduce(PyDUFuncObject * self, PyObject * args, PyObject *kws)
{
    return ufunc_dispatch.ufunc_reduce((PyObject*)self->ufunc, args, kws);
}

static PyObject *
dufunc_accumulate(PyDUFuncObject * self, PyObject * args, PyObject *kws)
{
    return ufunc_dispatch.ufunc_accumulate((PyObject*)self->ufunc, args, kws);
}

static PyObject *
dufunc_reduceat(PyDUFuncObject * self, PyObject * args, PyObject *kws)
{
    return ufunc_dispatch.ufunc_reduceat((PyObject*)self->ufunc, args, kws);
}

static PyObject *
dufunc_outer(PyDUFuncObject * self, PyObject * args, PyObject *kws)
{
    return ufunc_dispatch.ufunc_outer((PyObject*)self->ufunc, args, kws);
}


/*
 * The following are the vectorcall versions of the above, since NumPy
 * uses the FASTCALL/Vectorcall protocol starting with version 1.21.
 * The only NumPy versions supporting vectorcall use Python 3.7 or higher.
 */
static PyObject *
dufunc_reduce_fast(PyDUFuncObject * self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return ((_PyCFunctionFastWithKeywords)ufunc_dispatch.ufunc_reduce)(
            (PyObject*)self->ufunc, args, len_args, kwnames);
}

static PyObject *
dufunc_reduceat_fast(PyDUFuncObject * self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return ((_PyCFunctionFastWithKeywords)ufunc_dispatch.ufunc_reduceat)(
            (PyObject*)self->ufunc, args, len_args, kwnames);
}


static PyObject *
dufunc_accumulate_fast(PyDUFuncObject * self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return ((_PyCFunctionFastWithKeywords)ufunc_dispatch.ufunc_accumulate)(
            (PyObject*)self->ufunc, args, len_args, kwnames);
}


static PyObject *
dufunc_outer_fast(PyDUFuncObject * self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return ((_PyCFunctionFastWithKeywords)ufunc_dispatch.ufunc_outer)(
            (PyObject*)self->ufunc, args, len_args, kwnames);
}


static PyObject *
dufunc_at(PyDUFuncObject * self, PyObject * args)
{
    return ufunc_dispatch.ufunc_at((PyObject*)self->ufunc, args);
}

static PyObject *
dufunc__compile_for_args(PyDUFuncObject * self, PyObject * args,
                        PyObject * kws)
{
    PyErr_SetString(PyExc_NotImplementedError,
                    "Abstract method _DUFunc._compile_for_args() called!");
    return NULL;
}

static int *
_build_arg_types_array(PyObject * type_list, Py_ssize_t nargs)
{
    int *arg_types_array=NULL;
    Py_ssize_t idx, arg_types_size = PyList_Size(type_list);

    if (arg_types_size != nargs) {
        PyErr_SetString(
            PyExc_ValueError,
            "argument type list size does not equal ufunc argument count");
        return NULL;
    }
    arg_types_array = PyArray_malloc(sizeof(int) * nargs);
    if (!arg_types_array) {
        PyErr_NoMemory();
        return NULL;
    }
    for (idx = 0; idx < nargs; idx++) {
        arg_types_array[idx] = (int)PyLong_AsLong(PyList_GET_ITEM(type_list,
                                                                  idx));
    }
    if (PyErr_Occurred()) {
        PyArray_free(arg_types_array);
        arg_types_array = NULL;
    }
    return arg_types_array;
}

static PyObject *
dufunc__add_loop(PyDUFuncObject * self, PyObject * args)
{
    PyUFuncObject * ufunc=self->ufunc;
    void *loop_ptr=NULL, *data_ptr=NULL;
    int idx=-1, usertype=NPY_VOID;
    int *arg_types_arr=NULL;
    PyObject *arg_types=NULL, *loop_obj=NULL, *data_obj=NULL;

    if (self->frozen) {
        PyErr_SetString(PyExc_ValueError,
                        "_DUFunc._add_loop() called for frozen dufunc");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O!O!|O!",
                          &PyLong_Type, &loop_obj, &PyList_Type, &arg_types,
                          &PyLong_Type, &data_obj)) {
        return NULL;
    }

    loop_ptr = PyLong_AsVoidPtr(loop_obj);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (data_obj) {
        data_ptr = PyLong_AsVoidPtr(data_obj);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }

    arg_types_arr = _build_arg_types_array(arg_types, (Py_ssize_t)ufunc->nargs);
    if (!arg_types_arr) goto _dufunc__add_loop_fail;

    /* Check to see if any of the input types are user defined dtypes.
       If they are, we should use PyUFunc_RegisterLoopForType() since
       dispatch on a user defined dtype uses a Python dictionary
       keyed by usertype (and not the functions array).

       For more information, see how the usertype argument is used in
       PyUFunc_RegisterLoopForType(), defined by Numpy at
       .../numpy/core/src/umath/ufunc_object.c
    */
    for (idx = 0; idx < ufunc->nargs; idx++) {
        if (arg_types_arr[idx] >= NPY_USERDEF) {
            usertype = arg_types_arr[idx];
        }
    }

    if (usertype != NPY_VOID) {
        if (PyUFunc_RegisterLoopForType(ufunc, usertype,
                                        (PyUFuncGenericFunction)loop_ptr,
                                        arg_types_arr, data_ptr) < 0) {
            goto _dufunc__add_loop_fail;
        }
    } else {
        /* The following is an attempt to loosely follow the allocation
           code in Numpy.  See ufunc_frompyfunc() in
           .../numpy/core/src/umath/umathmodule.c.

           The primary goal is to allocate a single chunk of memory to
           hold the functions, data, and types loop arrays:

           ptr == |<- functions ->|<- data ->|<- types ->|

        */
        int ntypes=ufunc->ntypes + 1;
        PyUFuncGenericFunction *functions=NULL;
        void **data=NULL;
        char *types=NULL;
        void *newptr=NULL, *oldptr=NULL;
        size_t functions_size=sizeof(PyUFuncGenericFunction) * ntypes;
        size_t data_size=sizeof(void *) * ntypes;
        size_t type_ofs=sizeof(char) * ufunc->ntypes * ufunc->nargs;
        size_t newsize=(functions_size + data_size +
                        (sizeof(char) * ntypes * ufunc->nargs));

        oldptr = ufunc->ptr;
        newptr = PyArray_malloc(newsize);
        if (!newptr) {
            PyErr_NoMemory();
            goto _dufunc__add_loop_fail;
        }
        functions = (PyUFuncGenericFunction*)newptr;
        memcpy(functions, ufunc->functions,
               sizeof(PyUFuncGenericFunction) * ufunc->ntypes);
        functions[ntypes - 1] = (PyUFuncGenericFunction)loop_ptr;
        data = (void **)((char *)functions + functions_size);
        memcpy(data, ufunc->data, sizeof(void *) * ufunc->ntypes);
        data[ntypes - 1] = data_ptr;
        types = (char *)data + data_size;
        memcpy(types, ufunc->types, sizeof(char) * ufunc->ntypes *
               ufunc->nargs);
        for (idx = 0; idx < ufunc->nargs; idx++) {
            types[idx + type_ofs] = (char)arg_types_arr[idx];
        }

        ufunc->ntypes = ntypes;
        ufunc->functions = functions;
        ufunc->types = types;
        ufunc->data = data;
        ufunc->ptr = newptr;
        PyArray_free(oldptr);
    }

    PyArray_free(arg_types_arr);
    Py_INCREF(Py_None);
    return Py_None;

 _dufunc__add_loop_fail:
    PyArray_free(arg_types_arr);
    return NULL;
}

static struct PyMethodDef dufunc_methods[] = {
    {"reduce",
        (PyCFunction)dufunc_reduce,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"accumulate",
        (PyCFunction)dufunc_accumulate,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"reduceat",
        (PyCFunction)dufunc_reduceat,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"outer",
        (PyCFunction)dufunc_outer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"at",
        (PyCFunction)dufunc_at,
        METH_VARARGS, NULL},
    {"_compile_for_args",
        (PyCFunction)dufunc__compile_for_args,
        METH_VARARGS | METH_KEYWORDS,
        "Abstract method: subclasses should overload _compile_for_args() to compile the ufunc at the given arguments' types."},
    {"_add_loop",
        (PyCFunction)dufunc__add_loop,
        METH_VARARGS,
        NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};


/*
 * If Python is new enough, NumPy may use fastcall.  In that case we have to
 * also use fastcall for simplicity and speed.
 */
static struct PyMethodDef dufunc_methods_fast[] = {
    {"reduce",
        (PyCFunction)dufunc_reduce_fast,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"accumulate",
        (PyCFunction)dufunc_accumulate_fast,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"reduceat",
        (PyCFunction)dufunc_reduceat_fast,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"outer",
        (PyCFunction)dufunc_outer_fast,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"at",
        (PyCFunction)dufunc_at,
        METH_VARARGS, NULL},
    {"_compile_for_args",
        (PyCFunction)dufunc__compile_for_args,
        METH_VARARGS | METH_KEYWORDS,
        "Abstract method: subclasses should overload _compile_for_args() to compile the ufunc at the given arguments' types."},
    {"_add_loop",
        (PyCFunction)dufunc__add_loop,
        METH_VARARGS,
        NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};


static PyObject *
dufunc_getfrozen(PyDUFuncObject * self, void * closure)
{
    PyObject *result=(self->frozen) ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

static int
dufunc_setfrozen(PyDUFuncObject * self, PyObject * value, void * closure)
{
    int result=0;
    if (PyObject_IsTrue(value)) {
        self->frozen = 1;
    } else {
        PyErr_SetString(PyExc_ValueError,
                        "cannot clear the _DUFunc.frozen flag");
        result = -1;
    }
    return result;
}

static PyGetSetDef dufunc_getsets[] = {
    {"_frozen",
     (getter)dufunc_getfrozen, (setter)dufunc_setfrozen,
     "flag indicating call-time compilation has been disabled",
     NULL},
    {NULL}  /* Sentinel */
};

PyTypeObject PyDUFunc_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "numba._DUFunc",                            /* tp_name*/
    sizeof(PyDUFuncObject),                     /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    (destructor) dufunc_dealloc,                /* tp_dealloc */
    0,                                          /* tp_vectorcall_offset */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_as_async */
    (reprfunc) dufunc_repr,                     /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    (ternaryfunc) dufunc_call,                  /* tp_call */
    (reprfunc) dufunc_repr,                     /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    dufunc_methods,                             /* tp_methods */
    dufunc_members,                             /* tp_members */
    dufunc_getsets,                             /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc) dufunc_init,                     /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
    0,                                          /* tp_finalize */
    0,                                          /* tp_vectorcall */
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 12)
/* This was introduced first in 3.12
 * https://github.com/python/cpython/issues/91051
 */
    0,                                           /* tp_watched */
#endif
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 13)
/* This was introduced in 3.13
 * https://github.com/python/cpython/pull/114900
 */
    0,                                           /* tp_versions_used */
#endif

/* WARNING: Do not remove this, only modify it! It is a version guard to
 * act as a reminder to update this struct on Python version update! */
#if (PY_MAJOR_VERSION == 3)
#if ! (NB_SUPPORTED_PYTHON_MINOR)
#error "Python minor version is not supported."
#endif
#else
#error "Python major version is not supported."
#endif
/* END WARNING*/
};

/* ______________________________________________________________________
 * Module initialization boilerplate follows.
 */

static PyMethodDef ext_methods[] = {
    {"fromfunc", (PyCFunction) ufunc_fromfunc, METH_VARARGS, NULL},
    { NULL }
};

/* Don't remove this marker, it is used for inserting licensing code */
/*MARK1*/

MOD_INIT(_internal)
{
    PyObject *m;

    /* Don't remove this marker, it is used for inserting licensing code */
    /*MARK2*/

    import_array();
    import_umath();

    MOD_DEF(m, "_internal", "No docs",
            ext_methods)

    if (m == NULL)
        return MOD_ERROR_VAL;

    if (PyType_Ready(&PyUFuncCleaner_Type) < 0)
        return MOD_ERROR_VAL;

    PyDUFunc_Type.tp_new = PyType_GenericNew;

    int numpy_uses_fastcall = -1;
    if (init_ufunc_dispatch(&numpy_uses_fastcall) <= 0)
        return MOD_ERROR_VAL;

    if (numpy_uses_fastcall) {
        PyDUFunc_Type.tp_methods = dufunc_methods_fast;
    }

    if (PyType_Ready(&PyDUFunc_Type) < 0)
        return MOD_ERROR_VAL;
    Py_INCREF(&PyDUFunc_Type);
    if (PyModule_AddObject(m, "_DUFunc", (PyObject *)&PyDUFunc_Type) < 0)
        return MOD_ERROR_VAL;

    if (PyModule_AddIntMacro(m, PyUFunc_One)
        || PyModule_AddIntMacro(m, PyUFunc_Zero)
        || PyModule_AddIntMacro(m, PyUFunc_None)
        || PyModule_AddIntMacro(m, PyUFunc_ReorderableNone)
        )
        return MOD_ERROR_VAL;

    return MOD_SUCCESS_VAL(m);
}


#include "_ufunc.c"
