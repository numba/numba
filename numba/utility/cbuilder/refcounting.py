# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import *
from numba import llvm_types
from numba import typedefs
from numba.utility.cbuilder.library import register
from numba.utility.cbuilder.numbacdef import NumbaCDefinition, from_numba

from llvm_cbuilder import shortnames

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

p_py_ssize_t = shortnames.pointer(shortnames.py_ssize_t)

def ob_refcnt(obj_p):
    return deref(p_refcnt(obj_p))

def p_refcnt(obj_p):
    return obj_p.cast(p_py_ssize_t)

def deref(obj_p):
    return obj_p[0]

def const(ctemp, val):
    return ctemp.parent.constant(shortnames.py_ssize_t, val)

def add_refcnt(obj_p, refcnt):
    refcnt = const(obj_p, refcnt)
    refs = ob_refcnt(obj_p)
    refs += refcnt

def not_null(ptr):
    return ptr.cast(shortnames.py_ssize_t) != const(ptr, 0)

#------------------------------------------------------------------------
# Base Refcount Class
#------------------------------------------------------------------------

# TODO: Support tracing refcount operations (debug mode)
# TODO: Support refcount error checking (testing/debug mode)

class Refcounter(NumbaCDefinition):

    def set_signature(self, env, context):
        PyObject = typedefs.PyObject_HEAD

        self._argtys_ = [
            ('obj', PyObject.pointer().to_llvm(context)),
        ]
        self._retty_ = shortnames.void

#------------------------------------------------------------------------
# Refcount Implementations
#------------------------------------------------------------------------

@register
class Py_INCREF(Refcounter):
    "LLVM inline version of Py_INCREF"

    def body(self, obj):
        add_refcnt(obj, 1)
        self.ret()

@register
class Py_DECREF(Refcounter):
    "LLVM inline version of Py_DECREF"

    def body(self, obj):
        refcnt = ob_refcnt(obj)
        one = self.constant(refcnt.type, 1)
        Py_DecRef = self.external_cfunc('Py_DecRef')

        with self.ifelse(refcnt > one) as ifelse:

            with ifelse.then():
                # ob_refcnt > 1, just decrement
                add_refcnt(obj, -1)

            with ifelse.otherwise():
                # ob_refcnt == 1, dealloc
                Py_DecRef(obj)

        self.ret()

@register
class Py_XINCREF(Refcounter):
    "LLVM inline version of Py_XINCREF"

    def body(self, obj):
        with self.ifelse(not_null(obj)) as ifelse:
            with ifelse.then():
                add_refcnt(obj, 1)

        self.ret()

@register
class Py_XDECREF(Refcounter):
    "LLVM inline version of Py_XDECREF"

    def body(self, obj):
        py_decref = self.cbuilder_cfunc(Py_DECREF)

        with self.ifelse(not_null(obj)) as ifelse:
            with ifelse.then():
                py_decref(obj)

        self.ret()
