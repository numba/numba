'''
Implements basic vectorize
'''

from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import numpy as np

from . import _common

class BasicUFunc(CDefinition):
    '''a generic ufunc that wraps the workload
    '''
    _argtys_ = [
        ('args',       C.pointer(C.char_p)),
        ('dimensions', C.pointer(C.intp)),
        ('steps',      C.pointer(C.intp)),
        ('data',       C.void_p),
    ]

    def body(self, args, dimensions, steps, data,):
        ufunc_ptr = self.depends(self.FuncDef)
        fnty = ufunc_ptr.type.pointee

        with self.for_range(dimensions[0]) as (loop, item):
            _common.ufunc_core_impl(fnty, ufunc_ptr, args, steps, item)
        self.ret()

    @classmethod
    def specialize(cls, func_def):
        '''specialize to a workload
        '''
        cls._name_ = 'basicufunc_%s'% (func_def)
        cls.FuncDef = func_def

class _BasicVectorizeFromFunc(_common.CommonVectorizeFromFrunc):
    def build(self, lfunc):
        def_buf = BasicUFunc(CFuncRef(lfunc))
        return def_buf(lfunc.module)

basic_vectorize_from_func = _BasicVectorizeFromFunc()

class BasicVectorize(_common.GenericVectorize):
    def build_ufunc(self):
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        engine = self.translates[0]._get_ee()
        return basic_vectorize_from_func(lfunclist, engine=engine)


