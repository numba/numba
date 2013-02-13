import llvm.core

from numba import *

class RefcountingMixin(object):

    def decref(self, value, func='Py_DecRef'):
        "Py_DECREF a value"
        assert not self.nopython
        object_ltype = object_.to_llvm(self.context)
        b = self.builder
        mod = b.basic_block.function.module
        sig, py_decref = self.context.external_library.declare(mod, func)
        return b.call(py_decref, [b.bitcast(value, object_ltype)])

    def incref(self, value):
        "Py_INCREF a value"
        assert not self.nopython
        return self.decref(value, func='Py_IncRef')

    # TODO: generate efficient refcounting code, distinguish between dec/xdec
    xdecref = decref
    xincref = incref

    def xdecref_temp(self, temp, decref=None):
        "Py_XDECREF a temporary"
        assert not self.nopython
        decref = decref or self.decref
        decref(self.load_tbaa(temp, object_))

    def xincref_temp(self, temp):
        "Py_XINCREF a temporary"
        assert not self.nopython
        return self.xdecref_temp(temp, decref=self.incref)

    def xdecref_temp_cleanup(self, temp):
        "Cleanup a temp at the end of the function"

        assert not self.nopython
        bb = self.builder.basic_block

        self.builder.position_at_end(self.current_cleanup_bb)
        self.xdecref_temp(temp)
        self.current_cleanup_bb = self.builder.basic_block

        self.builder.position_at_end(bb)
