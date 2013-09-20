# -*- coding: utf-8 -*-
'''
Implements basic vectorize
'''
from __future__ import print_function, division, absolute_import
from llvm.core import *
from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import numpy as np

from . import _common

class BasicUFunc(CDefinition):
    '''a generic ufunc that wraps the workload
    '''
    _argtys_ = [
        ('args',       C.pointer(C.char_p), [ATTR_NO_ALIAS]),
        ('dimensions', C.pointer(C.intp), [ATTR_NO_ALIAS]),
        ('steps',      C.pointer(C.intp), [ATTR_NO_ALIAS]),
        ('data',       C.void_p, [ATTR_NO_ALIAS]),
    ]

    def body(self, args, dimensions, steps, data):
        ufunc_ptr = self.depends(self.FuncDef)
        fnty = ufunc_ptr.type.pointee

        # void return type implies return by reference in last argument
        is_void_ret = fnty.return_type.kind == lc.TYPE_VOID

        argtys = fnty.args[:-1] if is_void_ret else fnty.args

        arg_ptrs = []
        arg_steps = []
        for i in range(len(argtys)+1):
            arg_ptrs.append(self.var_copy(args[i]))
            const_steps = self.var_copy(steps[i])
            const_steps.invariant = True
            arg_steps.append(const_steps)

        with self.for_range(dimensions[0]) as (loop, item):
            callargs = []
            for i, argty in enumerate(argtys):
                if argty.kind == lc.TYPE_POINTER:
                    casted = arg_ptrs[i].cast(argty)
                    callargs.append(casted)
                else:
                    casted = arg_ptrs[i].cast(C.pointer(argty))
                    callargs.append(casted.load())
                # increment pointer
                arg_ptrs[i].assign(arg_ptrs[i][arg_steps[i]:])

            if is_void_ret:
                retty = fnty.args[-1]
                assert retty.kind == lc.TYPE_POINTER
                retval = self.var(retty.pointee)
                callargs.append(retval.ref)
                for i in callargs:
                    print(i)
                ufunc_ptr(*callargs, **dict(inline=True))
                res = retval
                retval_ptr = arg_ptrs[-1].cast(retty)
            else:
                res = ufunc_ptr(*callargs, **dict(inline=True))
                retval_ptr = arg_ptrs[-1].cast(C.pointer(fnty.return_type))
            retval_ptr.store(res, nontemporal=True)
            arg_ptrs[-1].assign(arg_ptrs[-1][arg_steps[-1]:])

        self.ret()

    def specialize(cls, func_def):
        '''specialize to a workload
        '''
        cls._name_ = 'basicufunc_%s'% (func_def)
        cls.FuncDef = func_def

class _BasicVectorizeFromFunc(_common.CommonVectorizeFromFunc):
    def build(self, lfunc, dtypes):
        def_buf = BasicUFunc(CFuncRef(lfunc))
        func = def_buf(lfunc.module)
        _common.post_vectorize_optimize(func)
        return func

basic_vectorize_from_func = _BasicVectorizeFromFunc()

class BasicASTVectorize(_common.GenericASTVectorize):

    _from_func_factory = basic_vectorize_from_func

    def build_ufunc(self, dispatcher=None):
        return self._from_func(dispatcher=dispatcher)

BasicVectorize = BasicASTVectorize
