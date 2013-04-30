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

    PyObject_HEAD = u.struct(pyobject_head_fields, 'PyObject_HEAD')
    PyArray = u.struct(pyobject_head_fields + [
         void_star,          # data
         u.int32,            # nd
         intp_star,          # dimensions
         intp_star,          # strides
         void_star,          # base
         void_star,          # descr
         u.int32,            # flags
         void_star,          # weakreflist
      ])

    PyCFunctionObject = u.struct([
        ('head', PyObject_HEAD),
        ('m_ml', void_star),
        ('m_self', u.object),
        ('m_module', u.object),
    ])

    # TODO: Parse C and Cython header files...
    NumbaFunctionObject = u.struct([
        ('pycfunction',         PyCFunctionObject),
        ('flags',               u.int),
        ('func_dict',           u.object),
        ('func_weakreflist',    u.object),
        ('func_name',           u.object),
        ('func_doc',            u.object),
        ('func_code',           u.object),
        ('func_closure',        u.object),
    ])

    return locals()

globals().update(define(defaults.numba_universe))