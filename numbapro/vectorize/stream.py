'''
Implements stream vectorize is a cache enhanced version of basic vectorize.

NOTE
----

Caching does not show any benefit. In fact, the code is running a lot slower.

'''

from llvm_cbuilder import *
import llvm_cbuilder.shortnames as C
import numpy as np
from .basic import BasicUFunc
from . import _common

class StreamUFunc(BasicUFunc):
    '''a generic ufunc that wraps the workload
    '''

    def body(self, args, dimensions, steps, data,):
        ufunc_ptr = self.depends(self.FuncDef)
        fnty = ufunc_ptr.type.pointee

        ZERO = self.constant(C.intp, 0)
        GRANUL = self.constant(C.intp, self.Granularity)

        caches = []
        arg_ptrs = []
        arg_steps = []

        const_count = self.var_copy(dimensions[0])
        const_count.invariant = True
        for i, ty in enumerate(fnty.args + [fnty.return_type]):
            arg_ptrs.append(self.var_copy(args[i]))
            const_step = self.var_copy(steps[i])
            const_step.invariant = True
            arg_steps.append(const_step)

            caches.append([])
            for j in range(self.Granularity):
                caches[i].append(self.var(ty))

        with self.for_range(ZERO, const_count, GRANUL) as (_, base):
            remain = self.min(const_count - base, GRANUL)
            with self.ifelse(GRANUL == remain) as ifelse:
                with ifelse.then():
                    # cache
                    for i, argty in enumerate(fnty.args):
                        for j in range(self.Granularity):
                            casted = arg_ptrs[i].cast(C.pointer(argty))
                            arg_val = casted.load()
                            caches[i][j].assign(arg_val)
                            arg_ptrs[i].assign(arg_ptrs[i][arg_steps[i]:])

                    # execute
                    for j in range(self.Granularity):
                        callargs = [caches[i][j] for i in range(len(fnty.args))]
                        caches[-1][j] = ufunc_ptr(*callargs, **dict(inline=True))

                    # writeback
                    for j in range(self.Granularity):
                        retval_ptr = arg_ptrs[-1].cast(C.pointer(fnty.return_type))
                        retval_ptr.store(caches[-1][j])
                        arg_ptrs[-1].assign(arg_ptrs[-1][arg_steps[-1]:])

                with ifelse.otherwise():
                    # skip caching for the remaining
                    for j in range(self.Granularity):
                        J = self.constant(remain.type, j)
                        callargs = []
                        with self.ifelse(J < remain) as ifelse:
                            with ifelse.then():
                                for i, argty in enumerate(fnty.args):
                                    casted = arg_ptrs[i].cast(C.pointer(argty))
                                    arg_val = casted.load()
                                    callargs.append(arg_val)
                                    arg_ptrs[i].assign(arg_ptrs[i][arg_steps[i]:])

                                res = ufunc_ptr(*callargs, **dict(inline=True))

                                retval_ptr = arg_ptrs[-1].cast(C.pointer(fnty.return_type))
                                retval_ptr.store(res, nontemporal=True)
                                arg_ptrs[-1].assign(arg_ptrs[-1][arg_steps[-1]:])

        self.ret()

    @classmethod
    def specialize(cls, func_def, granularity):
        '''specialize to a workload
        '''
        cls._name_ = 'streamufunc_%s'% (func_def)
        cls.Granularity = granularity
        cls.FuncDef = func_def

class _StreamVectorizeFromFunc(_common.CommonVectorizeFromFunc):
    def build(self, lfunc, dtypes, granularity):
        def_buf = StreamUFunc(CFuncRef(lfunc), granularity)
        func = def_buf(lfunc.module)

        _common.post_vectorize_optimize(func)

        return func

stream_vectorize_from_func = _StreamVectorizeFromFunc()

class StreamVectorize(_common.GenericVectorize):

    _from_func_factory = stream_vectorize_from_func

    def build_ufunc(self, granularity=8):
        return self._from_func(granularity=granularity)

    def build_ufunc_core(self, granularity=8):
        return super(StreamVectorize, self).build_ufunc_core(granularity=granularity)

class StreamASTVectorize(_common.GenericASTVectorize, StreamVectorize):
    pass