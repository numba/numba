/* Adapted from Cython/Utility/CythonFunction.c */

/* inline attribute */
#ifndef CYTHON_INLINE
  #if defined(__GNUC__)
    #define CYTHON_INLINE __inline__
  #elif defined(_MSC_VER)
    #define CYTHON_INLINE __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_INLINE inline
  #else
    #define CYTHON_INLINE
  #endif
#endif

/* unused attribute */
#ifndef CYTHON_UNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define CYTHON_UNUSED __attribute__ ((__unused__))
#   else
#     define CYTHON_UNUSED
#   endif
# elif defined(__ICC) || (defined(__INTEL_COMPILER) && !defined(_MSC_VER))
#   define CYTHON_UNUSED __attribute__ ((__unused__))
# else
#   define CYTHON_UNUSED
# endif
#endif

#if PY_VERSION_HEX < 0x02050000
  #define NAMESTR(n) ((char *)(n))
  #define DOCSTR(n)  ((char *)(n))
#else
  #define NAMESTR(n) (n)
  #define DOCSTR(n)  (n)
#endif

#include <Python.h>
#include <structmember.h>

#define CYFUNCTION_STATICMETHOD  0x01
#define CYFUNCTION_CLASSMETHOD   0x02
#define CYFUNCTION_CCLASS        0x04

#define CyFunction_GetClosure(f) \
    (((CyFunctionObject *) (f))->func_closure)
#define CyFunction_GetClassObj(f) \
    (((CyFunctionObject *) (f))->func_classobj)

#define CyFunction_Defaults(type, f) \
    ((type *)(((CyFunctionObject *) (f))->defaults))
#define CyFunction_SetDefaultsGetter(f, g) \
    ((CyFunctionObject *) (f))->defaults_getter = (g)


typedef struct {
    PyCFunctionObject func;
    int flags;
    PyObject *func_dict;
    PyObject *func_weakreflist;
    PyObject *func_name;
    PyObject *func_doc;
    PyObject *func_code;
    PyObject *func_closure;
    PyObject *func_classobj; /* No-args super() class cell */

    /* Dynamic default args*/
    void *defaults;
    int defaults_pyobjects;

    /* Defaults info */
    PyObject *defaults_tuple; /* Const defaults tuple */
    PyObject *(*defaults_getter)(PyObject *);
} CyFunctionObject;

static PyTypeObject *CyFunctionType = 0;

static PyObject *CyFunction_New(PyTypeObject *,
                                      PyMethodDef *ml, int flags,
                                      PyObject *self, PyObject *module,
                                      PyObject* code);

static CYTHON_INLINE void *CyFunction_InitDefaults(PyObject *m,
                                                   size_t size,
                                                   int pyobjects);
static CYTHON_INLINE void CyFunction_SetDefaultsTuple(PyObject *m,
                                                      PyObject *tuple);

/* Implementation */

static PyObject *
CyFunction_get_doc(CyFunctionObject *op, CYTHON_UNUSED void *closure)
{
    if (op->func_doc == NULL && op->func.m_ml->ml_doc) {
#if PY_MAJOR_VERSION >= 3
        op->func_doc = PyUnicode_FromString(op->func.m_ml->ml_doc);
#else
        op->func_doc = PyString_FromString(op->func.m_ml->ml_doc);
#endif
    }
    if (op->func_doc == 0) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    Py_INCREF(op->func_doc);
    return op->func_doc;
}

static int
CyFunction_set_doc(CyFunctionObject *op, PyObject *value)
{
    PyObject *tmp = op->func_doc;
    if (value == NULL)
        op->func_doc = Py_None; /* Mark as deleted */
    else
        op->func_doc = value;
    Py_INCREF(op->func_doc);
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
CyFunction_get_name(CyFunctionObject *op)
{
    if (op->func_name == NULL) {
#if PY_MAJOR_VERSION >= 3
        op->func_name = PyUnicode_InternFromString(op->func.m_ml->ml_name);
#else
        op->func_name = PyString_InternFromString(op->func.m_ml->ml_name);
#endif
    }
    Py_INCREF(op->func_name);
    return op->func_name;
}

static int
CyFunction_set_name(CyFunctionObject *op, PyObject *value)
{
    PyObject *tmp;

#if PY_MAJOR_VERSION >= 3
    if (value == NULL || !PyUnicode_Check(value)) {
#else
    if (value == NULL || !PyString_Check(value)) {
#endif
        PyErr_SetString(PyExc_TypeError,
                        "__name__ must be set to a string object");
        return -1;
    }
    tmp = op->func_name;
    Py_INCREF(value);
    op->func_name = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
CyFunction_get_self(CyFunctionObject *m, CYTHON_UNUSED void *closure)
{
    PyObject *self;

    self = m->func_closure;
    if (self == NULL)
        self = Py_None;
    Py_INCREF(self);
    return self;
}

static PyObject *
CyFunction_get_dict(CyFunctionObject *op)
{
    if (op->func_dict == NULL) {
        op->func_dict = PyDict_New();
        if (op->func_dict == NULL)
            return NULL;
    }
    Py_INCREF(op->func_dict);
    return op->func_dict;
}

static int
CyFunction_set_dict(CyFunctionObject *op, PyObject *value)
{
    PyObject *tmp;

    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError,
               "function's dictionary may not be deleted");
        return -1;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
               "setting function's dictionary to a non-dict");
        return -1;
    }
    tmp = op->func_dict;
    Py_INCREF(value);
    op->func_dict = value;
    Py_XDECREF(tmp);
    return 0;
}

/*
static PyObject *
CyFunction_get_globals(CYTHON_UNUSED CyFunctionObject *op)
{
    PyObject* dict = PyModule_GetDict(${module_cname});
    Py_XINCREF(dict);
    return dict;
}
*/

static PyObject *
CyFunction_get_closure(CYTHON_UNUSED CyFunctionObject *op)
{
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
CyFunction_get_code(CyFunctionObject *op)
{
    PyObject* result = (op->func_code) ? op->func_code : Py_None;
    Py_INCREF(result);
    return result;
}

static PyObject *
CyFunction_get_defaults(CyFunctionObject *op)
{
    if (op->defaults_tuple) {
        Py_INCREF(op->defaults_tuple);
        return op->defaults_tuple;
    }

    if (op->defaults_getter) {
        PyObject *res = op->defaults_getter((PyObject *) op);

        /* Cache result */
        if (res) {
            Py_INCREF(res);
            op->defaults_tuple = res;
        }
        return res;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyGetSetDef CyFunction_getsets[] = {
    {(char *) "func_doc", (getter)CyFunction_get_doc, (setter)CyFunction_set_doc, 0, 0},
    {(char *) "__doc__",  (getter)CyFunction_get_doc, (setter)CyFunction_set_doc, 0, 0},
    {(char *) "func_name", (getter)CyFunction_get_name, (setter)CyFunction_set_name, 0, 0},
    {(char *) "__name__", (getter)CyFunction_get_name, (setter)CyFunction_set_name, 0, 0},
    {(char *) "__self__", (getter)CyFunction_get_self, 0, 0, 0},
    {(char *) "func_dict", (getter)CyFunction_get_dict, (setter)CyFunction_set_dict, 0, 0},
    {(char *) "__dict__", (getter)CyFunction_get_dict, (setter)CyFunction_set_dict, 0, 0},
    /*{(char *) "func_globals", (getter)CyFunction_get_globals, 0, 0, 0},
    {(char *) "__globals__", (getter)CyFunction_get_globals, 0, 0, 0},*/
    {(char *) "func_closure", (getter)CyFunction_get_closure, 0, 0, 0},
    {(char *) "__closure__", (getter)CyFunction_get_closure, 0, 0, 0},
    {(char *) "func_code", (getter)CyFunction_get_code, 0, 0, 0},
    {(char *) "__code__", (getter)CyFunction_get_code, 0, 0, 0},
    {(char *) "func_defaults", (getter)CyFunction_get_defaults, 0, 0, 0},
    {(char *) "__defaults__", (getter)CyFunction_get_defaults, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

#ifndef PY_WRITE_RESTRICTED /* < Py2.5 */
#define PY_WRITE_RESTRICTED WRITE_RESTRICTED
#endif

static PyMemberDef CyFunction_members[] = {
    {(char *) "__module__", T_OBJECT, offsetof(CyFunctionObject, func.m_module), PY_WRITE_RESTRICTED, 0},
    {0, 0, 0,  0, 0}
};

static PyObject *
CyFunction_reduce(CyFunctionObject *m, CYTHON_UNUSED PyObject *args)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromString(m->func.m_ml->ml_name);
#else
    return PyString_FromString(m->func.m_ml->ml_name);
#endif
}

static PyMethodDef CyFunction_methods[] = {
    {NAMESTR("__reduce__"), (PyCFunction)CyFunction_reduce, METH_VARARGS, 0},
    {0, 0, 0, 0}
};


static PyObject *CyFunction_New(PyTypeObject *type, PyMethodDef *ml, int flags,
                                /*PyObject *closure,*/
                                PyObject *self, PyObject *module, PyObject* code)
{
    CyFunctionObject *op = PyObject_GC_New(CyFunctionObject, type);
    if (op == NULL)
        return NULL;
    op->flags = flags;
    op->func_weakreflist = NULL;
    op->func.m_ml = ml;
    /* op->func.m_self = (PyObject *) op;*/
    Py_XINCREF(self);
    op->func.m_self = self;
    /*Py_XINCREF(closure);
    op->func_closure = closure;*/
    op->func_closure = NULL;
    Py_XINCREF(module);
    op->func.m_module = module;
    op->func_dict = NULL;
    op->func_name = NULL;
    op->func_doc = NULL;
    op->func_classobj = NULL;
    Py_XINCREF(code);
    op->func_code = code;
    /* Dynamic Default args */
    op->defaults_pyobjects = 0;
    op->defaults = NULL;
    op->defaults_tuple = NULL;
    op->defaults_getter = NULL;
    PyObject_GC_Track(op);
    return (PyObject *) op;
}

PyObject *
CyFunction_NewEx(PyMethodDef *ml, int flags, PyObject *self,
                 PyObject *module, PyObject *code)
{
    return CyFunction_New(CyFunctionType, ml, flags, self, module, code);
}

static int
CyFunction_clear(CyFunctionObject *m)
{
    Py_CLEAR(m->func_closure);
    Py_CLEAR(m->func.m_module);
    Py_CLEAR(m->func_dict);
    Py_CLEAR(m->func_name);
    Py_CLEAR(m->func_doc);
    Py_CLEAR(m->func_code);
    Py_CLEAR(m->func_classobj);
    Py_CLEAR(m->defaults_tuple);

    if (m->defaults) {
        PyObject **pydefaults = CyFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_XDECREF(pydefaults[i]);

        PyMem_Free(m->defaults);
        m->defaults = NULL;
    }

    return 0;
}

static void CyFunction_dealloc(CyFunctionObject *m)
{
    PyObject_GC_UnTrack(m);
    if (m->func_weakreflist != NULL)
        PyObject_ClearWeakRefs((PyObject *) m);
    CyFunction_clear(m);
    PyObject_GC_Del(m);
}

static int CyFunction_traverse(CyFunctionObject *m, visitproc visit, void *arg)
{
    Py_VISIT(m->func_closure);
    Py_VISIT(m->func.m_module);
    Py_VISIT(m->func_dict);
    Py_VISIT(m->func_name);
    Py_VISIT(m->func_doc);
    Py_VISIT(m->func_code);
    Py_VISIT(m->func_classobj);
    Py_VISIT(m->defaults_tuple);

    if (m->defaults) {
        PyObject **pydefaults = CyFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_VISIT(pydefaults[i]);
    }

    return 0;
}

static PyObject *CyFunction_descr_get(PyObject *func, PyObject *obj, PyObject *type)
{
    CyFunctionObject *m = (CyFunctionObject *) func;

    if (m->flags & CYFUNCTION_STATICMETHOD) {
        Py_INCREF(func);
        return func;
    }

    if (m->flags & CYFUNCTION_CLASSMETHOD) {
        if (type == NULL)
            type = (PyObject *)(Py_TYPE(obj));
        return PyMethod_New(func,
                            type, (PyObject *)(Py_TYPE(type)));
    }

    if (obj == Py_None)
        obj = NULL;
    return PyMethod_New(func, obj, type);
}

static PyObject*
CyFunction_repr(CyFunctionObject *op)
{
    PyObject *func_name = CyFunction_get_name(op);

#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("<cyfunction %U at %p>",
                                func_name, (void *)op);
#else
    return PyString_FromFormat("<cyfunction %s at %p>",
                               PyString_AsString(func_name), (void *)op);
#endif
}

#if CYTHON_COMPILING_IN_PYPY
/* originally copied from PyCFunction_Call() in CPython's Objects/methodobject.c */
/* PyPy does not have this function */
static PyObject * CyFunction_Call(PyObject *func, PyObject *arg, PyObject *kw) {
    PyCFunctionObject* f = (PyCFunctionObject*)func;
    PyCFunction meth = PyCFunction_GET_FUNCTION(func);
    PyObject *self = PyCFunction_GET_SELF(func);
    Py_ssize_t size;

    switch (PyCFunction_GET_FLAGS(func) & ~(METH_CLASS | METH_STATIC | METH_COEXIST)) {
    case METH_VARARGS:
        if (likely(kw == NULL) || PyDict_Size(kw) == 0)
            return (*meth)(self, arg);
        break;
    case METH_VARARGS | METH_KEYWORDS:
        return (*(PyCFunctionWithKeywords)meth)(self, arg, kw);
    case METH_NOARGS:
        if (likely(kw == NULL) || PyDict_Size(kw) == 0) {
            size = PyTuple_GET_SIZE(arg);
            if (size == 0)
                return (*meth)(self, NULL);
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes no arguments (%zd given)",
                f->m_ml->ml_name, size);
            return NULL;
        }
        break;
    case METH_O:
        if (likely(kw == NULL) || PyDict_Size(kw) == 0) {
            size = PyTuple_GET_SIZE(arg);
            if (size == 1)
                return (*meth)(self, PyTuple_GET_ITEM(arg, 0));
            PyErr_Format(PyExc_TypeError,
                "%.200s() takes exactly one argument (%zd given)",
                f->m_ml->ml_name, size);
            return NULL;
        }
        break;
    default:
        PyErr_SetString(PyExc_SystemError, "Bad call flags in "
                        "CyFunction_Call. METH_OLDARGS is no "
                        "longer supported!");

        return NULL;
    }
    PyErr_Format(PyExc_TypeError, "%.200s() takes no keyword arguments",
                 f->m_ml->ml_name);
    return NULL;
}
#else
static PyObject * CyFunction_Call(PyObject *func, PyObject *arg, PyObject *kw) {
	return PyCFunction_Call(func, arg, kw);
}
#endif

static PyTypeObject CyFunctionType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    NAMESTR("cython_function_or_method"), /*tp_name*/
    sizeof(CyFunctionObject),   /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) CyFunction_dealloc, /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if PY_MAJOR_VERSION < 3
    0,                                  /*tp_compare*/
#else
    0,                                  /*reserved*/
#endif
    (reprfunc) CyFunction_repr,   /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    CyFunction_Call,              /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) CyFunction_traverse,   /*tp_traverse*/
    (inquiry) CyFunction_clear,   /*tp_clear*/
    0,                                  /*tp_richcompare*/
    offsetof(CyFunctionObject, func_weakreflist), /* tp_weaklistoffse */
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    CyFunction_methods,           /*tp_methods*/
    CyFunction_members,           /*tp_members*/
    CyFunction_getsets,           /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    CyFunction_descr_get,         /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    offsetof(CyFunctionObject, func_dict),/*tp_dictoffset*/
    0,                                  /*tp_init*/
    0,                                  /*tp_alloc*/
    0,                                  /*tp_new*/
    0,                                  /*tp_free*/
    0,                                  /*tp_is_gc*/
    0,                                  /*tp_bases*/
    0,                                  /*tp_mro*/
    0,                                  /*tp_cache*/
    0,                                  /*tp_subclasses*/
    0,                                  /*tp_weaklist*/
    0,                                  /*tp_del*/
#if PY_VERSION_HEX >= 0x02060000
    0,                                  /*tp_version_tag*/
#endif
};


int CyFunction_init(void) {
    // avoid a useless level of call indirection
    CyFunctionType_type.tp_call = PyCFunction_Call;
    if (PyType_Ready(&CyFunctionType_type) < 0)
        return -1;
    CyFunctionType = &CyFunctionType_type;
    return 0;
}

static CYTHON_INLINE void *CyFunction_InitDefaults(PyObject *func, size_t size, int pyobjects) {
    CyFunctionObject *m = (CyFunctionObject *) func;

    m->defaults = PyMem_Malloc(size);
    if (!m->defaults)
        return PyErr_NoMemory();
    memset(m->defaults, 0, sizeof(size));
    m->defaults_pyobjects = pyobjects;
    return m->defaults;
}

static CYTHON_INLINE void CyFunction_SetDefaultsTuple(PyObject *func, PyObject *tuple) {
    CyFunctionObject *m = (CyFunctionObject *) func;
    m->defaults_tuple = tuple;
    Py_INCREF(tuple);
}