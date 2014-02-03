# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba import *
from numba.typesystem import defaults

_trace_refs_ = hasattr(sys, 'getobjects')

def define(u):
    void_star = u.pointer(u.void)
    intp_star = u.pointer(u.npy_intp)

    if _trace_refs_:
        pyobject_head_extra_fields = [
            ('ob_next', void_star),
            ('ob_prev', void_star),
        ]
    else:
        pyobject_head_extra_fields = []

    pyobject_head_fields = pyobject_head_extra_fields + [
        ('ob_refcnt', u.Py_ssize_t),
        ('ob_type', void_star),
    ]

    PyObject_HEAD = u.struct_(pyobject_head_fields, 'PyObject_HEAD')
    PyArray = u.struct_(pyobject_head_fields + [
        ("data", void_star),
        ("nd", u.int32),
        ("dimensions", intp_star),
        ("strides", intp_star),
        ("base", void_star),
        ("descr", void_star),
        ("flags", u.int32),
        ("weakreflist", void_star),
      ])

    PyCFunctionObject = u.struct_([
        ('head', PyObject_HEAD),
        ('m_ml', void_star),
        ('m_self', u.object_),
        ('m_module', u.object_),
    ])

    # TODO: Parse C and Cython header files...
    NumbaFunctionObject = u.struct_([
        ('pycfunction',         PyCFunctionObject),
        ('flags',               u.int_),
        ('func_dict',           u.object_),
        ('func_weakreflist',    u.object_),
        ('func_name',           u.object_),
        ('func_doc',            u.object_),
        ('func_code',           u.object_),
        ('func_closure',        u.object_),
    ])

    return locals()

globals().update(define(defaults.numba_typesystem))