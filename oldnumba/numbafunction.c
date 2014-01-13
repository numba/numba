/* Adapted from Cython/Utility/CythonFunction.c */

#include <Python.h>
#include "_numba.h"

extern PyObject *Create_NumbaUnboundMethod(PyObject *, PyObject *);

#if PY_MAJOR_VERSION >= 3
  #define PyMethod_New(func, self, type) ( \
          (self) ? PyMethod_New(func, self) : \
                   Create_NumbaUnboundMethod(func, type))
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

#define NumbaFunction_STATICMETHOD  0x01
#define NumbaFunction_CLASSMETHOD   0x02
#define NumbaFunction_CCLASS        0x04

#define NumbaFunction_GetClosure(f) \
    (((NumbaFunctionObject *) (f))->func_closure)
#define NumbaFunction_GetClassObj(f) \
    (((NumbaFunctionObject *) (f))->func_classobj)

#define NumbaFunction_Defaults(type, f) \
    ((type *)(((NumbaFunctionObject *) (f))->defaults))
#define NumbaFunction_SetDefaultsGetter(f, g) \
    ((NumbaFunctionObject *) (f))->defaults_getter = (g)


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

    void *native_func;
    PyObject *native_signature;
    PyObject *keep_alive;

    /* Dynamic default args*/
    void *defaults;
    int defaults_pyobjects;

    /* Defaults info */
    PyObject *defaults_tuple; /* Const defaults tuple */
    PyObject *(*defaults_getter)(PyObject *);
} NumbaFunctionObject;

size_t closure_field_offset = offsetof(NumbaFunctionObject, func_closure);

PyTypeObject *NumbaFunctionType = 0;

static NumbaFunctionObject *NumbaFunction_New(PyTypeObject *type,
                                              PyMethodDef *ml, int flags,
                                              PyObject *closure,
                                              PyObject *self, PyObject *module,
                                              PyObject* code);

static NUMBA_INLINE void *NumbaFunction_InitDefaults(PyObject *m,
                                                   size_t size,
                                                   int pyobjects);
static NUMBA_INLINE void NumbaFunction_SetDefaultsTuple(PyObject *m,
                                                      PyObject *tuple);

/* Implementation */

static PyObject *
NumbaFunction_get_doc(NumbaFunctionObject *op, NUMBA_UNUSED void *closure)
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
NumbaFunction_set_doc(NumbaFunctionObject *op, PyObject *value)
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
NumbaFunction_get_name(NumbaFunctionObject *op)
{
    if (op->func_name == NULL) {
#if PY_MAJOR_VERSION >= 3
        op->func_name = PyUnicode_InternFromString(op->func.m_ml->ml_name);
#else
        op->func_name = PyString_InternFromString(op->func.m_ml->ml_name);
#endif
    }
    Py_XINCREF(op->func_name);
    return op->func_name;
}

static int
NumbaFunction_set_name(NumbaFunctionObject *op, PyObject *value)
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
NumbaFunction_get_self(NumbaFunctionObject *m, NUMBA_UNUSED void *closure)
{
    PyObject *self;

    self = m->func_closure;
    if (self == NULL)
        self = Py_None;
    Py_INCREF(self);
    return self;
}

static PyObject *
NumbaFunction_get_dict(NumbaFunctionObject *op)
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
NumbaFunction_set_dict(NumbaFunctionObject *op, PyObject *value)
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
NumbaFunction_get_globals(NUMBA_UNUSED NumbaFunctionObject *op)
{
    PyObject* dict = PyModule_GetDict(${module_cname});
    Py_XINCREF(dict);
    return dict;
}
*/

static PyObject *
NumbaFunction_get_closure(NUMBA_UNUSED NumbaFunctionObject *op)
{
    if (op->func_closure) {
        Py_INCREF(op->func_closure);
        return op->func_closure;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *
NumbaFunction_get_code(NumbaFunctionObject *op)
{
    PyObject* result = (op->func_code) ? op->func_code : Py_None;
    Py_INCREF(result);
    return result;
}

static PyObject *
NumbaFunction_get_defaults(NumbaFunctionObject *op)
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

static PyGetSetDef NumbaFunction_getsets[] = {
    {(char *) "func_doc", (getter)NumbaFunction_get_doc, (setter)NumbaFunction_set_doc, 0, 0},
    {(char *) "__doc__",  (getter)NumbaFunction_get_doc, (setter)NumbaFunction_set_doc, 0, 0},
    {(char *) "func_name", (getter)NumbaFunction_get_name, (setter)NumbaFunction_set_name, 0, 0},
    {(char *) "__name__", (getter)NumbaFunction_get_name, (setter)NumbaFunction_set_name, 0, 0},
    {(char *) "__self__", (getter)NumbaFunction_get_self, 0, 0, 0},
    {(char *) "func_dict", (getter)NumbaFunction_get_dict, (setter)NumbaFunction_set_dict, 0, 0},
    {(char *) "__dict__", (getter)NumbaFunction_get_dict, (setter)NumbaFunction_set_dict, 0, 0},
    /*{(char *) "func_globals", (getter)NumbaFunction_get_globals, 0, 0, 0},
    {(char *) "__globals__", (getter)NumbaFunction_get_globals, 0, 0, 0},*/
    {(char *) "func_closure", (getter)NumbaFunction_get_closure, 0, 0, 0},
    {(char *) "__closure__", (getter)NumbaFunction_get_closure, 0, 0, 0},
    {(char *) "func_code", (getter)NumbaFunction_get_code, 0, 0, 0},
    {(char *) "__code__", (getter)NumbaFunction_get_code, 0, 0, 0},
    {(char *) "func_defaults", (getter)NumbaFunction_get_defaults, 0, 0, 0},
    {(char *) "__defaults__", (getter)NumbaFunction_get_defaults, 0, 0, 0},
    {0, 0, 0, 0, 0}
};

#ifndef PY_WRITE_RESTRICTED /* < Py2.5 */
#define PY_WRITE_RESTRICTED WRITE_RESTRICTED
#endif

static PyMemberDef NumbaFunction_members[] = {
    {(char *) "__module__", T_OBJECT, offsetof(NumbaFunctionObject, func.m_module), PY_WRITE_RESTRICTED, 0},
    {0, 0, 0,  0, 0}
};

static PyObject *
NumbaFunction_reduce(NumbaFunctionObject *m, NUMBA_UNUSED PyObject *args)
{
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromString(m->func.m_ml->ml_name);
#else
    return PyString_FromString(m->func.m_ml->ml_name);
#endif
}

static PyMethodDef NumbaFunction_methods[] = {
    {NAMESTR("__reduce__"), (PyCFunction)NumbaFunction_reduce, METH_VARARGS, 0},
    {0, 0, 0, 0}
};


static NumbaFunctionObject *NumbaFunction_New(
            PyTypeObject *type, PyMethodDef *ml, int flags, PyObject *closure,
            PyObject *module, PyObject *code, PyObject *keep_alive)
{
    NumbaFunctionObject *op = PyObject_GC_New(NumbaFunctionObject, type);
    if (op == NULL)
        return NULL;
    op->flags = flags;
    op->func_weakreflist = NULL;
    op->func.m_ml = ml;
    op->func.m_self = (PyObject *) op; /* No incref or decref here */
    Py_XINCREF(closure);
    op->func_closure = closure;
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

    Py_XINCREF(keep_alive);
    op->keep_alive = keep_alive;
    op->native_func = NULL;
    op->native_signature = NULL;

    PyObject_GC_Track((PyObject *)op);
    return op;
}

/* Create a new function and set the closure scope */
PyObject *
NumbaFunction_NewEx(PyMethodDef *ml, PyObject *module, PyObject *code,
                    PyObject *closure, void *native_func,
                    PyObject *native_signature, PyObject *keep_alive)
{
    NumbaFunctionObject *result = NumbaFunction_New(
                        NumbaFunctionType, ml, 0, closure, module,
                        code, keep_alive);
    if (result) {
        result->native_func = native_func;
        Py_XINCREF(native_signature);
        result->native_signature = native_signature;
    }
    return (PyObject *)result;
}

static int
NumbaFunction_clear(NumbaFunctionObject *m)
{
    Py_CLEAR(m->func_closure);
    Py_CLEAR(m->func.m_module);
    Py_CLEAR(m->func_dict);
    Py_CLEAR(m->func_name);
    Py_CLEAR(m->func_doc);
    Py_CLEAR(m->func_code);
    Py_CLEAR(m->func_classobj);
    Py_CLEAR(m->defaults_tuple);
    Py_CLEAR(m->keep_alive);

    if (m->defaults) {
        PyObject **pydefaults = NumbaFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_XDECREF(pydefaults[i]);

        PyMem_Free(m->defaults);
        m->defaults = NULL;
    }

    return 0;
}

static void NumbaFunction_dealloc(NumbaFunctionObject *m)
{
    PyObject_GC_UnTrack(m);
    if (m->func_weakreflist != NULL)
        PyObject_ClearWeakRefs((PyObject *) m);
    NumbaFunction_clear(m);
    PyObject_GC_Del(m);
}

static int NumbaFunction_traverse(NumbaFunctionObject *m, visitproc visit, void *arg)
{
    Py_VISIT(m->func_closure);
    Py_VISIT(m->func.m_module);
    Py_VISIT(m->func_dict);
    Py_VISIT(m->func_name);
    Py_VISIT(m->func_doc);

    Py_VISIT(m->func_code);
    Py_VISIT(m->func_classobj);
    Py_VISIT(m->defaults_tuple);

    Py_VISIT(m->keep_alive);

    if (m->defaults) {
        PyObject **pydefaults = NumbaFunction_Defaults(PyObject *, m);
        int i;

        for (i = 0; i < m->defaults_pyobjects; i++)
            Py_VISIT(pydefaults[i]);
    }

    return 0;
}

static PyObject *NumbaFunction_descr_get(PyObject *func, PyObject *obj, PyObject *type)
{
    NumbaFunctionObject *m = (NumbaFunctionObject *) func;

    if (m->flags & NumbaFunction_STATICMETHOD) {
        Py_INCREF(func);
        return func;
    }

    if (m->flags & NumbaFunction_CLASSMETHOD) {
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
NumbaFunction_repr(NumbaFunctionObject *op)
{
    /* Py_RETURN_NONE; */
    PyObject *func_name = NumbaFunction_get_name(op);

#if PY_MAJOR_VERSION >= 3
    return PyUnicode_FromFormat("<NumbaFunction %U at %p>",
                                func_name, (void *)op);
#else
    return PyString_FromFormat("<NumbaFunction %s at %p>",
                               PyString_AsString(func_name), (void *)op);
#endif
}

static PyTypeObject NumbaFunctionType_type = {
    PyVarObject_HEAD_INIT(0, 0)
    NAMESTR("numba_function_or_method"), /*tp_name*/
    sizeof(NumbaFunctionObject),   /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor) NumbaFunction_dealloc, /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
#if PY_MAJOR_VERSION < 3
    0,                                  /*tp_compare*/
#else
    0,                                  /*reserved*/
#endif
    (reprfunc) NumbaFunction_repr,   /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash*/
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags*/
    0,                                  /*tp_doc*/
    (traverseproc) NumbaFunction_traverse,   /*tp_traverse*/
    (inquiry) NumbaFunction_clear,   /*tp_clear*/
    0,                                  /*tp_richcompare*/
    offsetof(NumbaFunctionObject, func_weakreflist), /* tp_weaklistoffse */
    0,                                  /*tp_iter*/
    0,                                  /*tp_iternext*/
    NumbaFunction_methods,           /*tp_methods*/
    NumbaFunction_members,           /*tp_members*/
    NumbaFunction_getsets,           /*tp_getset*/
    0,                                  /*tp_base*/
    0,                                  /*tp_dict*/
    NumbaFunction_descr_get,         /*tp_descr_get*/
    0,                                  /*tp_descr_set*/
    offsetof(NumbaFunctionObject, func_dict),/*tp_dictoffset*/
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


int NumbaFunction_init(void) {
    // avoid a useless level of call indirection
    NumbaFunctionType_type.tp_call = PyCFunction_Call;
    if (PyType_Ready(&NumbaFunctionType_type) < 0)
        return -1;
    NumbaFunctionType = &NumbaFunctionType_type;
    return 0;
}

static NUMBA_INLINE void *NumbaFunction_InitDefaults(PyObject *func, size_t size, int pyobjects) {
    NumbaFunctionObject *m = (NumbaFunctionObject *) func;

    m->defaults = PyMem_Malloc(size);
    if (!m->defaults)
        return PyErr_NoMemory();
    memset(m->defaults, 0, sizeof(size));
    m->defaults_pyobjects = pyobjects;
    return m->defaults;
}

static NUMBA_INLINE void NumbaFunction_SetDefaultsTuple(PyObject *func, PyObject *tuple) {
    NumbaFunctionObject *m = (NumbaFunctionObject *) func;
    m->defaults_tuple = tuple;
    Py_INCREF(tuple);
}
