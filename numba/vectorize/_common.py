# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
import llvm.core as lc

import numba
from numba import decorators
from numba.codegen.llvmcontext import LLVMContextManager
from . import _internal

try:
    ptr_t = long
except NameError:
    ptr_t = int
    assert False, "Have not check this yet" # Py3.0?

_llvm_ty_str_to_numpy = {
            'i8'     : np.int8,
            'i16'    : np.int16,
            'i32'    : np.int32,
            'i64'    : np.int64,
            'float'  : np.float32,
            'double' : np.float64,
        }

def _llvm_ty_to_numpy(ty):
    return _llvm_ty_str_to_numpy[str(ty)]

def _llvm_ty_to_dtype(ty):
    return np.dtype(_llvm_ty_to_numpy(ty)) #.num

_numbatypes_str_to_numpy = {
            'int8'     : np.int8,
            'int16'    : np.int16,
            'int32'    : np.int32,
            'int64'    : np.int64,
            'uint8'    : np.uint8,
            'uint16'   : np.uint16,
            'uint32'   : np.uint32,
            'uint64'   : np.uint64,
#            'f'        : np.float32,
#            'd'        : np.float64,
            'float'    : np.float32,
            'double'   : np.float64,
        }

def _numbatypes_to_numpy(ty):
    # ret = _numbatypes_str_to_numpy[str(ty)]
    return ty.get_dtype()

class CommonVectorizeFromFunc(object):
    def build(self, lfunc, dtypes):
        raise NotImplementedError

    def get_dtype_nums(self, tyslist):
        return [[dtype.num for dtype in dtypes] for dtypes in tyslist]

    def __call__(self, lfunclist, tyslist, engine,
                 dispatcher=None,
                 **kws):
        '''create ufunc from a llvm.core.Function

        lfunclist : a single or iterable of llvm.core.Function instance
        engine : a llvm.ee.ExecutionEngine instance

        return a function object which can be called from python.
        '''
        try:
            iter(lfunclist)
        except TypeError:
            lfunclist = [lfunclist]

        ptrlist = self._prepare_pointers(lfunclist, tyslist, engine, **kws)

        fntype = lfunclist[0].type.pointee
        inct = len(fntype.args)
        if fntype.return_type.kind == lc.TYPE_VOID:
            # void return type means return value is passed as argument.
            inct -= 1
        outct = 1

        datlist = [None] * len(lfunclist)

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)
        tyslist = self.get_dtype_nums(tyslist)
        ufunc = _internal.fromfunc(ptrlist, tyslist, inct, outct,
                                   datlist, dispatcher)
        return ufunc

    def _prepare_ufunc_core(self, lfunclist, tyslist, **kws):
        spuflist = []
        for i, (lfunc, dtypes) in enumerate(zip(lfunclist, tyslist)):
            spuflist.append(self.build(lfunc, dtypes, **kws))
        return spuflist

    def _get_pointer_from_ufunc_core(self, spuf, engine):
        fptr = engine.get_pointer_to_function(spuf)
        return ptr_t(fptr)

    def _prepare_pointers(self, lfunclist, tyslist, engine, **kws):
        spuflist = self._prepare_ufunc_core(lfunclist, tyslist, **kws)
        ptrlist = [self._get_pointer_from_ufunc_core(spuf, engine)
                   for spuf in spuflist]
        return ptrlist

    def _prepare_prototypes_and_pointers(self, lfunclist, tyslist, engine, **kws):
        spuflist = self._prepare_ufunc_core(lfunclist, tyslist, **kws)
        ptrlist = [self._get_pointer_from_ufunc_core(spuf, engine)
                   for spuf in spuflist]
        return zip(spuflist, ptrlist)

class GenericASTVectorize(object):

    def __init__(self, func):
        self.pyfunc = func
        self.translates = []
        self.args_restypes = getattr(self, 'args_restypes', [])
        self.signatures = []
        self.llvm_context = LLVMContextManager()

    def _get_tys_list(self):
        types_lists = []
        for numba_func in self.translates:
            dtype_nums = []
            types_lists.append(dtype_nums)
            for arg_type in self.get_argtypes(numba_func):
                dtype = arg_type.get_dtype()
                dtype_nums.append(dtype)

        return types_lists

    def _get_lfunc_list(self):
        return [t.lfunc for t in self.translates]

    def _get_ee(self):
        return self.llvm_context.execution_engine

    def build_ufunc(self):
        raise NotImplementedError

    def register_ufunc(self, ufunc):
        from numba.type_inference.modules import numpyufuncs
        numpyufuncs.register_arbitrary_ufunc(ufunc)

    def _from_func_factory(self, lfunclist, tyslist, **kws):
        """
        Set this attribute to some subclass of CommonVectorizeFromFunc
        """
        raise NotImplementedError

    def _from_func(self, **kws):
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        tyslist = self._get_tys_list()
        engine = self._get_ee()
        ufunc = self._from_func_factory(lfunclist, tyslist, engine=engine, **kws)
        self.register_ufunc(ufunc)
        return ufunc

    def build_ufunc_core(self, **kws):
        '''Build the ufunc core functions and returns the prototype and pointer.
        The return value is a list of tuples (prototype, pointer).
        '''
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        tyslist = self._get_tys_list()
        engine = self._get_ee()
        get_proto_ptr = self._from_func_factory._prepare_prototypes_and_pointers
        return get_proto_ptr(lfunclist, tyslist, engine, **kws)

    def add(self, restype=None, argtypes=None, **kwds):
        """
        Add a specialization to the vectorizer. Pass any keyword arguments
        to numba.jit().
        """
        kwds['ctypes'] = True
        dec = decorators.jit(restype, argtypes, **kwds)
        numba_func = dec(self.pyfunc)
        self.args_restypes.append(list(numba_func.signature.args) +
                                  [numba_func.signature.return_type])
        self.signatures.append((restype, argtypes, {}))
        self.translates.append(numba_func)

    def get_argtypes(self, numba_func):
        return list(numba_func.signature.args) + [numba_func.signature.return_type]


def post_vectorize_optimize(func):
    '''Perform aggressive optimization after each vectorizer.
        
    TODO: review if this is necessary.
    '''
    cm = LLVMContextManager()
    return cm.pass_manager


