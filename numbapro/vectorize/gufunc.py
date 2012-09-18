from . import _common
from numba import *
import numba.decorators
from numba.minivect import minitypes
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from numbapro.translate import Translate
from numbapro import _internal, _cudadispatch
import numpy as np

class _GeneralizedUFuncFromFrunc(_common.CommonVectorizeFromFrunc):
    def build_functions(self, ptrlist):
        """
        Return a list of pointers to the kernels.
        """
        return ptrlist

    def __call__(self, lfunclist, tyslist, signature, engine, use_cuda, **kws):
        '''create generailized ufunc from a llvm.core.Function

        lfunclist : a single or iterable of llvm.core.Function instance
        engine : a llvm.ee.ExecutionEngine instance

        return a function object which can be called from python.
        '''
        kws['signature'] = signature

        try:
            iter(lfunclist)
        except TypeError:
            lfunclist = [lfunclist]

        ptrlist = self._prepare_pointers(lfunclist, engine, **kws)
        self.build_functions(ptrlist)

        inct = len(tyslist[0]) - 1
        outct = 1

        datlist = [None] * len(lfunclist)

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)

        gufunc = _internal.fromfuncsig(ptrlist, tyslist, inct, outct, datlist,
                                       signature, use_cuda)

        return gufunc

    def build(self, lfunc, signature):
        def_guf = GUFuncEntry(signature, CFuncRef(lfunc))
        guf = def_guf(lfunc.module)
        # print guf
        return guf

class _GeneralizedUFuncFromFrunc(_GeneralizedUFuncFromFrunc):
    def build_functions(self, ptrlist):
        return _cudadispatch.build_cuda_functions()

class GUFuncVectorize(object):

    gufunc_from_func = _GeneralizedUFuncFromFrunc()

    def __init__(self, func, sig):
        self.pyfunc = func
        self.translates = []
        self.signature = sig

    def add(self, arg_types):
        t = Translate(self.pyfunc, arg_types=arg_types)
        t.translate()
        self.translates.append(t)

    def _get_tys_list(self):
        from numba.translate import convert_to_llvmtype
        tyslist = []
        for t in self.translates:
            tys = []
            for ty in t.arg_types:
                while isinstance(ty, list):
                    ty = ty[0]
                lty = convert_to_llvmtype(ty)
                tys.append(np.dtype(_common._llvm_ty_to_numpy(lty)).num)
            tyslist.append(tys)
        return tyslist

    def _get_lfunc_list(self):
        return [t.lfunc for t in self.translates]

    def build_ufunc(self, use_cuda=False):
        assert self.translates, "No translation"
        lfunclist = self._get_lfunc_list()
        tyslist = self._get_tys_list()
        engine = self.translates[0]._get_ee()
        return self.gufunc_from_func(
            lfunclist, tyslist, self.signature, engine, use_cuda=use_cuda)

class CUDAGUFuncVectorize(GUFuncVectorize):
    def build_ufunc(self):
        return super(CUDAGUFuncVectorize, self).build_ufunc(use_cuda=True)


class PyObjectHead(CStruct):
    _fields_ = [
        ('f1', C.intp),
        ('f2', C.pointer(C.int)),
        ]

_intp_ptr = C.pointer(C.intp)

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
        ('maskna_dtype',   C.void_p),
        ('maskna_data',    C.void_p),
        ('maskna_strides', _intp_ptr),
    ]

    def fakeit(self, data, dimensions, steps):
        assert len(dimensions) == len(steps)
        self.data.assign(data)
        self.nd.assign(self.parent.constant(C.int, len(dimensions)))

        ary_dims = self.parent.array(C.intp, len(dimensions))
        for i, dim in enumerate(dimensions):
            ary_dims[i] = dim

        self.dimensions.assign(ary_dims)

        ary_steps = self.parent.array(C.intp, len(steps))
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

    def _outer_loop(self, dimensions, innerfunc, pyarys, steps):
        # implement outer loop
        with self.for_range(dimensions[0]) as (loop, idx):
            innerfunc(*map(lambda x: x.reference(), pyarys))

            for i, ary in enumerate(pyarys):
                ary.data.assign(ary.data[steps[i]:])

    def body(self, args, dimensions, steps, data):
        innerfunc = self.depends(self.FuncDef)

        diminfo = list(_parse_signature(self.Signature))
        n_pyarys = len(diminfo)

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
        for i, ary in enumerate(pyarys):
            ary_ndim = len(diminfo[i])
            ary_dims = [dimensions[1 + dims.index(k)] for k in diminfo[i]]
            ary_steps = []

            for j in range(ary_ndim):
                ary_steps.append(steps[step_offset])
                step_offset += 1

            ary.fakeit(args[i], ary_dims, ary_steps)

        self._outer_loop(args, dimensions, innerfunc, pyarys, steps)
        self.ret()

    @classmethod
    def specialize(cls, signature, func_def):
        '''specialize to a workload
        '''
        signature = signature.replace(' ', '') # remove all spaces
        cls._name_ = 'gufunc_%s_%s'% (signature, func_def)
        cls.FuncDef = func_def
        cls.Signature = signature

def get_cuda_outer_loop(builder):
    context = numba.decorators.context

    arg_types = [
        char.pointer().pointer(), # char **args
        npy_intp.pointer(),       # npy_intp *dimensions
        npy_intp.pointer(),       # npy_intp *steps
        void.pointer(),           # void *func
        object_.pointer(),        # PyObject **arrays
    ]
    signature = minitypes.FunctionTYpe(return_type=void, arg_types=arg_types)
    lfunc_type = signature.pointer().to_llvm(numba.decorators.context)

    func_addr = _cudadispatch.get_cuda_outer_loop_addr()
    func_int_addr = llvm.core.Constant.int(int64.to_llvm(context), func_addr)
    func_pointer = builder.inttoptr(func_int_addr, lfunc_type)
    lfunc = builder.load(func_pointer)
    return CFuncRef(lfunc)

def create_wrapper(kernel, nops):
    class CUDAKernelWrapper(CDefinition):
        """
        Wrapper around generalized ufunc that computes the data pointer for each
        array.
        """

        _argtys_ = [('op%d', PyArray) for i in range(nops)]
        _argtys_ += [('steps', C.npy_intp_p)]

        def body(self, *arrays, **kwds):
            steps = kwds.pop('step')

            # get current thread index
            tid_x = self.get_intrinsic(INTR_PTX_READ_TID_X, [])
            ntid_x = self.get_intrinsic(INTR_PTX_READ_NTID_X, [])
            ctaid_x = self.get_intrinsic(INTR_PTX_READ_CTAID_X, [])

            tid = self.var_copy(tid_x())
            blkdim = self.var_copy(ntid_x())
            blkid = self.var_copy(ctaid_x())

            # Adjust data pointer
            i = steps[tid]
            for array in arrays:
                array.data.assign(array.data[i:])

            # Call actual kernel
            kernel(*arrays)

            self.ret()

        @classmethod
        def specialize(cls, signature, func_def):
            '''specialize to a workload
            '''
            signature = signature.replace(' ', '') # remove all spaces
            cls._name_ = 'gufunc_kernel_wrapper_%s_%s'% (signature, func_def)
            cls.FuncDef = func_def
            cls.Signature = signature

class GUFuncCUDAEntry(GUFuncEntry):

    def _outer_loop(self, dimensions, innerfunc, pyarys, steps):
        super(GUFuncCUDAEntry, self)._outer_loop(dimensions, innerfunc,
                                                 pyarys, steps)
        cuda_outer_loop = get_cuda_outer_loop(innerfunc.parent.builder)
        cuda_outer_loop(args, dimensions, steps, innerfunc, pyarys)
