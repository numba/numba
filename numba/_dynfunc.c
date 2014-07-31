#include "_pymodule.h"

#include <string.h>

/* NOTE: EnvironmentObject and ClosureObject must be kept in sync with
 * the definitions in numba/targets/base.py (EnvBody and ClosureBody).
 */

/*
 * EnvironmentObject hosts data needed for execution of compiled functions.
 * For now, it is only used in object mode (though it still gets passed
 * to nopython functions).
 */
typedef struct {
    PyObject_HEAD
    PyObject *globals;
    /* Assorted "constants" that are needed at runtime to execute
       the compiled function.  This can include frozen closure variables,
       lifted loops, etc. */
    PyObject *consts;
} EnvironmentObject;


static PyMemberDef env_members[] = {
    {"globals", T_OBJECT, offsetof(EnvironmentObject, globals), READONLY},
    {"consts", T_OBJECT, offsetof(EnvironmentObject, consts), READONLY},
    {NULL}  /* Sentinel */
};

static int
env_traverse(EnvironmentObject *env, visitproc visit, void *arg)
{
    Py_VISIT(env->globals);
    Py_VISIT(env->consts);
    return 0;
}

static void
env_dealloc(EnvironmentObject *env)
{
    _PyObject_GC_UNTRACK((PyObject *) env);
    Py_DECREF(env->globals);
    Py_DECREF(env->consts);
    Py_TYPE(env)->tp_free((PyObject *) env);
}

static PyObject *
env_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyObject *globals;
    EnvironmentObject *env;
    static char *kwlist[] = {"globals", 0};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!:function", kwlist,
            &PyDict_Type, &globals))
        return NULL;

    env = (EnvironmentObject *) PyType_GenericNew(type, args, kwds);
    if (env == NULL)
        return NULL;
    Py_INCREF(globals);
    env->globals = globals;
    env->consts = PyList_New(0);
    if (!env->consts) {
        Py_DECREF(env);
        return NULL;
    }
    return (PyObject *) env;
}


static PyTypeObject EnvironmentType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_dynfunc.Environment",   /*tp_name*/
    sizeof(EnvironmentObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor) env_dealloc,  /*tp_dealloc*/
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
    (traverseproc) env_traverse, /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    env_members,               /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    env_new,                   /* tp_new */
};

/* A closure object is created for each call to make_function(), and stored
   as the resulting PyCFunction object's "self" pointer.  It points to an
   EnvironmentObject which is constructed during compilation.  This allows
   for two things:
       - lifetime management of dependent data (e.g. lifted loop dispatchers)
       - access to the execution environment by the compiled function
         (for example the globals module)
   */

typedef struct {
    PyObject_HEAD
    /* The dynamically-filled method definition for the PyCFunction object
       using this closure. */
    PyMethodDef def;
    EnvironmentObject *env;
    PyObject *weakreflist;
    /* We could also store the LLVM function or engine here, to ensure
       generated code is kept alive. */
} ClosureObject;


static int
closure_traverse(ClosureObject *clo, visitproc visit, void *arg)
{
    Py_VISIT(clo->env);
    return 0;
}

static void
closure_dealloc(ClosureObject *clo)
{
    _PyObject_GC_UNTRACK((PyObject *) clo);
    if (clo->weakreflist != NULL)
        PyObject_ClearWeakRefs((PyObject *) clo);
    PyObject_Free((void *) clo->def.ml_name);
    PyObject_Free((void *) clo->def.ml_doc);
    Py_XDECREF(clo->env);
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
    offsetof(ClosureObject, weakreflist), /* tp_weaklistoffset */
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
    EnvironmentObject *env;

    if (!PyArg_ParseTuple(args, "OOOOO!",
            &module, &fname, &fdoc, &fnaddrobj, &EnvironmentType, &env)) {
        return NULL;
    }

    fnaddr = PyLong_AsVoidPtr(fnaddrobj);
    if (fnaddr == NULL && PyErr_Occurred())
        return NULL;

    closure = closure_new(module, fname, fdoc, fnaddr);
    if (closure == NULL)
        return NULL;
    Py_INCREF(env);
    closure->env = env;

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
    PyObject *m, *impl_info;

    MOD_DEF(m, "_dynfunc", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    if (PyType_Ready(&ClosureType))
        return MOD_ERROR_VAL;
    if (PyType_Ready(&EnvironmentType))
        return MOD_ERROR_VAL;

    impl_info = Py_BuildValue(
        "{snsn}",
        "offset_closure_body", offsetof(ClosureObject, env),
        "offset_env_body", offsetof(EnvironmentObject, globals)
        );
    if (impl_info == NULL)
        return MOD_ERROR_VAL;
    PyModule_AddObject(m, "_impl_info", impl_info);

    Py_INCREF(&ClosureType);
    PyModule_AddObject(m, "_Closure", (PyObject *) (&ClosureType));
    Py_INCREF(&EnvironmentType);
    PyModule_AddObject(m, "Environment", (PyObject *) (&EnvironmentType));

    return MOD_SUCCESS_VAL(m);
}
