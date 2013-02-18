from numba import *

_trace_refs_ = hasattr(sys, 'getobjects')

if _trace_refs_:
    pyobject_head_extra_fields = [
        ('ob_next', void.pointer()),
        ('ob_prev', void.pointer()),
    ]
else:
    pyobject_head_extra_fields = []

pyobject_head_fields = pyobject_head_extra_fields + [
    ('ob_refcnt', Py_ssize_t),
    ('ob_type', void.pointer()),
]

PyObject_HEAD = struct(pyobject_head_fields, 'PyObject_HEAD')

PyCFunctionObject = struct([
    ('head', PyObject_HEAD),
    ('m_ml', void.pointer()),
    ('m_self', object_),
    ('m_module', object_),
])

"""
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
"""

# TODO: Parse C and Cython header files...
NumbaFunctionObject = struct([
    ('pycfunction',         PyCFunctionObject),
    ('flags',               int_),
    ('func_dict',           object_),
    ('func_weakreflist',    object_),
    ('func_name',           object_),
    ('func_doc',            object_),
    ('func_code',           object_),
    ('func_closure',        object_),
])