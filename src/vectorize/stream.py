'''
Implements stream vectorize is a cache enhanced version of basic vectorize
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

        # populate steps for cache
        cache_steps = self.array(C.intp, len(fnty.args) + 1)
        for i, arg in enumerate(fnty.args):
            cache_steps[i] = self.sizeof(arg)
        cache_steps[len(fnty.args)] = self.sizeof(fnty.return_type)

        #populate cache

        cache_ptr = []
        for arg in fnty.args:  # cache for arguments
            cache_ptr.append(self.array(arg, GRANUL))
        cache_ptr.append(self.array(fnty.return_type, GRANUL))  # cache for results


        cache = self.array(C.void_p, len(cache_ptr))
        for i, ptr in enumerate(cache_ptr):
            cache[i] = ptr.cast(C.void_p)


        with self.for_range(ZERO, dimensions[0], GRANUL) as (outer, base):
            get_offset = lambda B, I, S, T: B[I * S].reference().cast(C.pointer(T))

            remain = self.min(dimensions[0] - base, GRANUL)
            with self.for_range(remain) as (inner, offset): # do cache
                for i, arg in enumerate(fnty.args):
                    cache_ptr[i][offset] = get_offset(args[i], base+offset, steps[i], arg).load()

            with self.for_range(remain) as (inner, offset): # do work
                _common.ufunc_core_impl(fnty, ufunc_ptr, cache, cache_steps, offset)

            with self.for_range(remain) as (inner, offset): # extract result
                outptr = get_offset(args[len(fnty.args)], base+offset,
                                    steps[len(fnty.args)], fnty.return_type)
                outptr.store(cache_ptr[len(fnty.args)][offset])
        self.ret()

    @classmethod
    def specialize(cls, func_def, granularity):
        '''specialize to a workload
        '''
        cls._name_ = 'streamufunc_%s'% (func_def)
        cls.Granularity = granularity
        cls.FuncDef = func_def

class _StreamVectorizeFromFunc(_common.CommonVectorizeFromFrunc):
    def build(self, lfunc, granularity):
        def_buf = StreamUFunc(CFuncRef(lfunc), granularity)
        func = def_buf(lfunc.module)
        return func

stream_vectorize_from_func = _StreamVectorizeFromFunc()

class StreamVectorize(_common.GenericVectorize):
    def build_ufunc(self, granularity=32):
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        engine = self.translates[0]._get_ee()
        return stream_vectorize_from_func(lfunclist, engine=engine,
                                          granularity=granularity)


