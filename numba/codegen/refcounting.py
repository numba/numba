# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

from numba import *
from numba.utility.cbuilder import refcounting

class RefcountingMixin(object):

    def refcount(self, func, value):
        "Refcount a value with a refcounting function"
        assert not self.nopython

        refcounter = self.context.cbuilder_library.declare(func, self.env,
                                                           self.llvm_module)
        object_ltype = object_.to_llvm(self.context)

        b = self.builder
        return b.call(refcounter, [b.bitcast(value, object_ltype)])

    def decref(self, value):
        "Py_DECREF a value"
        return self.refcount(refcounting.Py_DECREF, value)

    def incref(self, value):
        "Py_INCREF a value"
        return self.refcount(refcounting.Py_INCREF, value)

    def xdecref(self, value):
        "Py_XDECREF a value"
        return self.refcount(refcounting.Py_XDECREF, value)

    def xincref(self, value):
        "Py_XINCREF a value"
        return self.refcount(refcounting.Py_XINCREF, value)

    def xdecref_temp(self, temp):
        "Py_XDECREF a temporary"
        return self.xdecref(self.load_tbaa(temp, object_))

    def xincref_temp(self, temp):
        "Py_XINCREF a temporary"
        return self.xincref(self.load_tbaa(temp, object_))

    def xdecref_temp_cleanup(self, temp):
        """
        Cleanup a temp at the end of the function:

            * Save current basic block
            * Generate code at cleanup path
            * Restore basic block
        """
        assert not self.nopython
        bb = self.builder.basic_block

        self.builder.position_at_end(self.current_cleanup_bb)
        self.xdecref_temp(temp)
        self.current_cleanup_bb = self.builder.basic_block

        self.builder.position_at_end(bb)
