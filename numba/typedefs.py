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
