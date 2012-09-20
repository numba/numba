from . import _common

from numba import *
from numba import llvm_types
import numba.decorators
from numba.minivect import minitypes

from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C
from llvm_cbuilder import builder
from numbapro.translate import Translate
from numbapro import _internal, _cudadispatch
from numbapro.vectorize import cuda

import numpy as np
import llvm.core

class _GeneralizedUFuncFromFunc(_common.CommonVectorizeFromFrunc):
    def datalist(self, lfunclist, ptrlist, cuda_dispatcher):
        """
        Return a list of data pointers to the kernels.
        """
        return [None] * len(lfunclist)

    def __call__(self, lfunclist, tyslist, signature, engine, use_cuda,
                 cuda_dispatcher=None, **kws):
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
        inct = len(tyslist[0]) - 1
        outct = 1
        datlist = self.datalist(lfunclist, ptrlist, cuda_dispatcher)

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)

        gufunc = _internal.fromfuncsig(ptrlist, tyslist, inct, outct, datlist,
                                       signature, cuda_dispatcher)

        return gufunc

    def build(self, lfunc, signature):
        def_guf = GUFuncEntry(signature, CFuncRef(lfunc))
        guf = def_guf(lfunc.module)
        # print guf
        return guf


class GUFuncVectorize(object):
    """
    Vectorizer for generalized ufuncs.
    """

    gufunc_from_func = _GeneralizedUFuncFromFunc()

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

    def _outer_loop(self, args, dimensions, innerfunc, pyarys, steps, data):
        # implement outer loop
        innerfunc = self.depends(self.FuncDef)
        with self.for_range(dimensions[0]) as (loop, idx):
            innerfunc(*map(lambda x: x.reference(), pyarys))

            for i, ary in enumerate(pyarys):
                ary.data.assign(ary.data[steps[i]:])

    def body(self, args, dimensions, steps, data):
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

        self._outer_loop(args, dimensions, pyarys, steps, data)
        self.ret()

    @classmethod
    def specialize(cls, signature, func_def):
        '''specialize to a workload
        '''
        signature = signature.replace(' ', '') # remove all spaces
        cls._name_ = 'gufunc_%s_%s'% (signature, func_def)
        cls.FuncDef = func_def
        cls.Signature = signature

#
### Generalized CUDA ufuncs
#

class _GeneralizedCUDAUFuncFromFunc(_GeneralizedUFuncFromFunc):

    def __init__(self, module, signature):
        self.module = module
        # Create a wrapper around _cuda.c:cuda_outer_loop
        wrapper_builder = GUFuncCUDAEntry(signature, None)
        self.wrapper = wrapper_builder(self.module)
        self.cuda_kernels = None

    def datalist(self, lfunclist, ptrlist, cuda_dispatcher):
        """
        Build a bunch of CudaFunctionAndData and make sure it is passed to
        our ufunc.
        """
        func_names = [lfunc.name for lfunc in self.cuda_kernels]
        return cuda_dispatcher.build_datalist(func_names)

    def build(self, lfunc, signature):
        """
        lfunc: lfunclist was [wrapper] * n_funcs, so we're done
        """
        #return lfunc
        wrapper_builder = GUFuncCUDAEntry(signature, None)
        return wrapper_builder(self.module)


class CudaVectorize(cuda.CudaVectorize):
    """
    Builds a wrapper for generalized ufunc CUDA kernels.
    """

    def __init__(self, func):
        super(CudaVectorize, self).__init__(func)
        self.cuda_wrappers = []

    def _build_caller(self, lfunc):
        assert self.module is lfunc.module

        lfunc.calling_convention = llvm.core.CC_PTX_DEVICE
        lfunc.linkage = llvm.core.LINKAGE_INTERNAL # do not emit device function
        lcaller_def = create_kernel_wrapper(lfunc)
        lcaller = lcaller_def(self.module)
        lcaller.calling_convention = llvm.core.CC_PTX_KERNEL
        self.cuda_wrappers.append(lcaller)
        return lcaller


class CUDAGUFuncVectorize(GUFuncVectorize):
    """
    Generalized ufunc vectorizer. Executes generalized ufuncs on the GPU.
    """

    def __init__(self, func, sig):
        super(CUDAGUFuncVectorize, self).__init__(func, sig)
        self.cuda_vectorizer = CudaVectorize(func)
        self.llvm_module = llvm.core.Module.new('default_module')
        self.llvm_ee = llvm.ee.EngineBuilder.new(
                    self.llvm_module).force_jit().opt(3).create()
        self.gufunc_from_func = _GeneralizedCUDAUFuncFromFunc(
                                            self.llvm_module, sig)
        # self.llvm_fpm = llvm.passes.FunctionPassManager.new(self.llvm_module)
        # self.llvm_fpm.initialize()

    def add(self, arg_types):
        self.cuda_vectorizer.add(ret_type=void, arg_types=arg_types)

    def _get_tys_list(self):
        types = []
        for ret_type, arg_types, kwargs in self.cuda_vectorizer.signatures:
            tys = arg_types + [ret_type]
            types.append([
                minitypes.map_minitype_to_dtype(t.dtype if t.is_array else t).num
                    for t in arg_types])

        return types

    def build_ufunc(self, device_number=-1):
        n_funcs = len(self.cuda_vectorizer.signatures)
        lfunclist = [self.gufunc_from_func.wrapper] * n_funcs
        tyslist = self._get_tys_list()
        dispatcher = self.cuda_vectorizer._build_ufunc(device_number)
        self.gufunc_from_func.cuda_kernels = self.cuda_vectorizer.cuda_wrappers
        return self.gufunc_from_func(
            lfunclist, tyslist, self.signature, engine=self.llvm_ee,
            cuda_dispatcher=dispatcher, use_cuda=True)


wrapper_count = 0
def create_kernel_wrapper(kernel):
    class CUDAKernelWrapper(CDefinition):
        """
        Wrapper around generalized ufunc that computes the data pointer for
        each array on the GPU.
        """

        _name_ = 'cuda_wrapper%d' % wrapper_count
        wrapper_count += 1
        _retty_ = C.void
        _argtys_ = [('op%d' % i, llvm.core.Type.pointer(PyArray.llvm_type()))
                        for i in range(len(kernel.args))]
        _argtys_ += [('steps', C.npy_intp_p)]

        def body(self, *args):
            arrays = args[:-1]
            steps = args[-1]

            # get current thread index
            tid_x = self.get_intrinsic(llvm.core.INTR_PTX_READ_TID_X, [])
            ntid_x = self.get_intrinsic(llvm.core.INTR_PTX_READ_NTID_X, [])
            ctaid_x = self.get_intrinsic(llvm.core.INTR_PTX_READ_CTAID_X, [])

            tid = self.var_copy(tid_x())
            blkdim = self.var_copy(ntid_x())
            blkid = self.var_copy(ctaid_x())

            # Adjust data pointer for the kernel
            # Note that we invoke the kernel N times simultaneously, so to
            # adjust the data pointer we need a copy of each PyArrayObject
            # struct
            b = steps.parent.builder
            _int32_zero = llvm.core.Constant.int(llvm_types._int32, 0)
            id = tid + blkdim * blkid
            arrays = [self.var_copy(array) for array in arrays]

            for i, array in enumerate(arrays):
                # array = PyArray(self, array.handle)
                # offset = id * steps[i].cast(id.type)
                # array.data.assign(array.data[offset:])

                array = array.handle
                offset = id * steps[i].cast(id.type)
                loffset = offset.handle
                ldata_pointer = b.gep(array, [_int32_zero])
                b.store(b.gep(b.load(ldata_pointer), [loffset]),
                        ldata_pointer)

            # Call actual kernel
            # kernel_func = self.depends(CFuncRef(kernel))
            # kernel_func(*arrays)
            b.call(kernel, [array.handle for array in arrays])

            self.ret()

        @classmethod
        def specialize(cls):
            pass

    return CUDAKernelWrapper()


def get_cuda_outer_loop(builder):
    """
    Build an llvm_func that references _cuda.c:cuda_outer_loop
    """
    context = numba.decorators.context

    arg_types = [
        char.pointer().pointer(), # char **args
        npy_intp.pointer(),       # npy_intp *dimensions
        npy_intp.pointer(),       # npy_intp *steps
        void.pointer(),           # void *func
        object_.pointer(),        # PyObject **arrays
    ]
    signature = minitypes.FunctionType(return_type=void, args=arg_types)
    lfunc_type = signature.pointer().to_llvm(numba.decorators.context)

    func_addr = _cudadispatch.get_cuda_outer_loop_addr()
    # return CFuncRef('cuda_outer_loop', lfunc_type, func_addr)
    func_int_addr = llvm.core.Constant.int(int64.to_llvm(context), func_addr)
    func_pointer = builder.inttoptr(func_int_addr, lfunc_type)
    lfunc = func_pointer
    # lfunc = builder.load(func_pointer)
    # return CFuncRef(func_pointer)
    return lfunc

num_wrappers = 0
class GUFuncCUDAEntry(GUFuncEntry):
    """
    This function is invoked by NumPy and sets up the fake PyArrayObjects and
    calls _cuda.c:cuda_outer_loop
    """

    def _outer_loop(self, args, dimensions, py_arrays, steps, info):
        """
        innerfunc: the CUDA kernel
        """
        llvm_builder = args.parent.builder
        cbuilder = args.parent

        cuda_outer_loop = get_cuda_outer_loop(llvm_builder)
        array_list = cbuilder.array(llvm_types._numpy_array, len(py_arrays))
        for i, py_array in enumerate(py_arrays):
            # array_list[i].assign(py_array.ref)
            llvm_builder.store(py_array.handle, array_list[i].handle)

        largs = [llvm_builder.load(arg.handle) for arg in (args, dimensions,
                                                           steps, info)]
        largs.append(array_list.handle)
        llvm_builder.call(cuda_outer_loop, largs)

    @classmethod
    def specialize(cls, signature, func_def):
        '''specialize to a workload
        '''
        global num_wrappers
        super(GUFuncCUDAEntry, cls).specialize(signature, func_def)
        cls._name_ = 'cuda_outer_loop_wrapper_%d' % num_wrappers
        num_wrappers += 1

