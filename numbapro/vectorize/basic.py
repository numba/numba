'''
Implements basic vectorize
'''
from llvm.core import *
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

        arg_ptrs = []
        arg_steps = []
        for i in range(len(fnty.args)+1):
            arg_ptrs.append(self.var_copy(args[i]))
            arg_steps.append(self.var_copy(steps[i]))

        with self.for_range(dimensions[0]) as (loop, item):
            callargs = []
            for i, argty in enumerate(fnty.args):
                casted = arg_ptrs[i].cast(C.pointer(argty))
                callargs.append(casted.load())
                arg_ptrs[i].assign(arg_ptrs[i][arg_steps[i]:]) # increment pointer

            res = ufunc_ptr(*callargs, **dict(inline=True))
            retval_ptr = arg_ptrs[-1].cast(C.pointer(fnty.return_type))
            retval_ptr.store(res)
            arg_ptrs[-1].assign(arg_ptrs[-1][arg_steps[-1]:])
            #arg_ptrs[-1].assign(arg_ptrs[-1][arg_steps[-1]:], nontemporal=False)

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
        func = def_buf(lfunc.module)

        _common.post_vectorize_optimize(func)

        return func

basic_vectorize_from_func = _BasicVectorizeFromFunc()

class BasicVectorize(_common.GenericVectorize):
    def build_ufunc(self, minivect_dispatcher=None):
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        tyslist = self._get_tys_list()
        engine = self.translates[0]._get_ee()
        return basic_vectorize_from_func(lfunclist, tyslist, engine=engine,
                                         minivect_dispatcher=minivect_dispatcher)


