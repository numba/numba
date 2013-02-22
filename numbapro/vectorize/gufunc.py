from . import _common

from numba import *
from numba import llvm_types

from llvm.core import Type, inline_function, ATTR_NO_ALIAS, ATTR_NO_CAPTURE
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder
# from numba.vectorize._translate import Translate
# from numba.vectorize import _internal
from _numba_vectorize._translate import Translate
from _numba_vectorize import _internal

import numpy as np


class _GeneralizedUFuncFromFunc(_common.CommonVectorizeFromFunc):
    def datalist(self, lfunclist, ptrlist):
        """
        Return a list of data pointers to the kernels.
        """
        return [None] * len(lfunclist)

    def __call__(self, lfunclist, tyslist, signature, engine,
                 vectorizer, **kws):
        '''create generailized ufunc from a llvm.core.Function

        lfunclist : a single or iterable of llvm.core.Function instance
        engine : a llvm.ee.ExecutionEngine instance

        return a function object which can be called from python.
        '''
        assert 'cuda_dispatcher' not in kws, "Temporary check for mismatch API"
        kws['signature'] = signature

        try:
            iter(lfunclist)
        except TypeError:
            lfunclist = [lfunclist]

        self.tyslist = tyslist
        ptrlist = self._prepare_pointers(lfunclist, tyslist, engine, **kws)
        inct = len(tyslist[0]) - 1
        outct = 1
        datlist = self.datalist(lfunclist, ptrlist)

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)

        # Hold on to the vectorizer while the ufunc lives
        tyslist = self.get_dtype_nums(tyslist)
        gufunc = _internal.fromfuncsig(ptrlist, tyslist, inct, outct, datlist,
                                       signature, vectorizer)

        return gufunc

    def build(self, lfunc, dtypes, signature):
        def_guf = GUFuncEntry(dtypes, signature, CFuncRef(lfunc))
        guf = def_guf(lfunc.module)
        # print guf
        return guf


class GUFuncVectorize(object):
    """
    Vectorizer for generalized ufuncs.
    """

    def __init__(self, func, sig):
        self.pyfunc = func
        self.translates = []
        self.signature = sig
        self.gufunc_from_func = _GeneralizedUFuncFromFunc()

    def add(self, argtypes, restype=void):
        assert restype == void or restype is None
        t = Translate(self.pyfunc, argtypes=argtypes)
        t.translate()
        self.translates.append(t)

    def _get_tys_list(self):
        from numba.translate import convert_to_llvmtype
        tyslist = []
        for t in self.translates:
            tys = []
            for ty in t.argtypes:
                while isinstance(ty, list):
                    ty = ty[0]
                lty = convert_to_llvmtype(ty)
                tys.append(np.dtype(_common._llvm_ty_to_numpy(lty)))
            tyslist.append(tys)
        return tyslist

    def _get_lfunc_list(self):
        return [t.lfunc for t in self.translates]

    def _get_ee(self):
        return self.translates[0]._get_ee()

    def build_ufunc(self):
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        tyslist = self._get_tys_list()
        engine = self._get_ee()
        return self.gufunc_from_func(
            lfunclist, tyslist, self.signature, engine,
            vectorizer=self)

class GUFuncASTVectorize(_common.ASTVectorizeMixin, GUFuncVectorize):
    "Use the AST numba backend to compile the gufunc"

    def get_argtypes(self, numba_func):
        return numba_func.signature.args

_intp_ptr = C.pointer(C.intp)

class PyObjectHead(CStruct):
    _fields_ = [
        ('ob_refcnt', C.intp),
        # NOTE: not a integer, just need to match definition in numba
        ('ob_type', C.pointer(C.int)),
    ]

    if llvm_types._trace_refs_:
        # Account for _PyObject_HEAD_EXTRA
        _fields_ = [
            ('ob_next', _intp_ptr),
            ('ob_prev', _intp_ptr),
        ] + _fields_


class PyArray(CStruct):

    _fields_ = PyObjectHead._fields_ + [
        ('data',           C.void_p),
        ('nd',             C.int),
        ('dimensions',     _intp_ptr),
        ('strides',        _intp_ptr),
        ('base',           C.void_p),
        ('descr',          C.void_p),
        ('flags',          C.int),
        ('weakreflist',    C.void_p),
#        ('maskna_dtype',   C.void_p),
#        ('maskna_data',    C.void_p),
#        ('maskna_strides', _intp_ptr),
    ]

    def fakeit(self, dtype, data, dimensions, steps):
        assert len(dimensions) == len(steps)
        constant = self.parent.constant

        self.ob_refcnt.assign(constant(C.intp, 1))
        type_p = constant(C.py_ssize_t, id(np.ndarray))
        self.ob_type.assign(type_p.cast(C.pointer(C.int)))

        self.base.assign(self.parent.constant_null(C.void_p))
        dtype_p = constant(C.py_ssize_t, id(dtype))
        self.descr.assign(dtype_p.cast(C.void_p))
        self.flags.assign(constant(C.int, _internal.NPY_WRITEABLE))

        self.data.assign(data)
        self.nd.assign(constant(C.int, len(dimensions)))

        ary_dims = self.parent.array(C.intp, len(dimensions) * 2)
        ary_steps = ary_dims[len(dimensions):]
        for i, dim in enumerate(dimensions):
            ary_dims[i] = dim

        self.dimensions.assign(ary_dims)

        # ary_steps = self.parent.array(C.intp, len(steps))
        for i, step in enumerate(steps):
            ary_steps[i] = step
        self.strides.assign(ary_steps)


def _parse_signature(sig):
    inargs, outarg = sig.split('->')

    for inarg in filter(bool, inargs.split(')')):
        dimnames = inarg[1+inarg.find('('):].split(',')
        yield dimnames
    else:
        dimnames = outarg.strip('()').split(',')
        yield dimnames

class GUFuncEntry(CDefinition):
    '''a generalized ufunc that wraps a numba jit'ed function

    NOTE: Currently, this only works for array return type.
    And, return type must be the last argument of the nubma jit'ed function.
    '''
    _argtys_ = [
        ('args',       C.pointer(C.char_p)),
        ('dimensions', C.pointer(C.intp)),
        ('steps',      C.pointer(C.intp)),
        ('data',       C.void_p),
    ]

    def _outer_loop(self, args, dimensions, pyarys, steps, data):
        # implement outer loop
        innerfunc = self.depends(self.FuncDef)
        with self.for_range(dimensions[0]) as (loop, idx):
            args = [arg.reference().cast(arg_type.type)
                        for arg, arg_type in zip(pyarys, innerfunc.handle.args)]
            innerfunc(*args)
            # innerfunc(*map(lambda x: x.reference(), pyarys))

            for i, ary in enumerate(pyarys):
                ary.data.assign(ary.data[steps[i]:])

    def body(self, args, dimensions, steps, data):
        diminfo = list(_parse_signature(self.Signature))
        n_pyarys = len(diminfo)
        assert n_pyarys == len(self.dtypes)

        # extract unique dimension names
        dims = []
        for grp in diminfo:
            for it in grp:
                if it not in dims:
                    dims.append(it)

        # build pyarrays for argument to inner function
        pyarys = [self.var(PyArray) for _ in range(n_pyarys)]

        # populate pyarrays
        step_offset = len(pyarys)
        for i, (dtype, ary) in enumerate(zip(self.dtypes, pyarys)):
            ary_ndim = len(diminfo[i])
            ary_dims = [dimensions[1 + dims.index(k)] for k in diminfo[i]]
            ary_steps = []

            for j in range(ary_ndim):
                ary_steps.append(steps[step_offset])
                step_offset += 1

            ary.fakeit(dtype, args[i], ary_dims, ary_steps)

        self._outer_loop(args, dimensions, pyarys, steps, data)
        self.ret()

    @classmethod
    def specialize(cls, dtypes, signature, func_def):
        '''specialize to a workload
        '''
        signature = signature.replace(' ', '') # remove all spaces
        cls.dtypes = dtypes
        cls._name_ = 'gufunc_%s_%s'% (signature, func_def)
        cls.FuncDef = func_def
        cls.Signature = signature
