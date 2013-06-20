import numpy as np
from numbapro.npm import types
from numbapro.npm.typing import (cast_penalty, Restrict, MustBe,
                                 Conditional, int_set)
from numbapro.npm.errors import CompileError
from . import ptx
import numbapro

class CudaPyInferError(CompileError):
    def __init__(self, value, msg):
        super(CudaPyInferError, self).__init__(value.lineno, msg)

def cast_from_sregtype(value):
    return cast_penalty(ptx.SREG_TYPE, value)

def rule_sreg(infer, value, obj):
    infer.rules[value].add(Conditional(cast_from_sregtype))
    infer.rules[value].add(Restrict(int_set))
    value.replace(value=obj)

def rule_grid_macro(infer, value):
    args = value.args.args
    if len(args) != 1:
        raise CudaPyInferError(value, "grid() takes exactly 1 argument")
    arg = args[0].value
    if arg.kind != 'Const':
        raise CudaPyInferError(value, "arg to grid() must be a constant")
    ndim = arg.args.value
    if ndim not in [1, 2]:
        raise CudaPyInferError(value, "arg to grid() must be 1 or 2")

    if ndim == 1:
        infer.rules[value].add(Conditional(cast_from_sregtype))
        infer.rules[value].add(Restrict(int_set))
    else:
        assert ndim == 2
        tuplety = types.tupletype(ptx.SREG_TYPE, 2)
        infer.rules[value].add(MustBe(tuplety))
        infer.possible_types.add(tuplety)

    value.replace(func=ptx.grid)

def rule_syncthread(infer, value):
    if value.args.args:
        raise CudaPyInferError(value, "syncthreads() takes no arguments")
    value.replace(func=ptx.syncthreads)

def rule_np_dtype(infer, value, obj):
    value.replace(value=np.dtype(obj))
    #dtype = np.dtype(value.args.value)
    #nptype = UserType(str(dtype), dtype)
    #infer.possible_types.add(nptype) # should not increase type set

NP_DTYPE_MAP = {
    np.dtype(np.int8):          types.int8,
    np.dtype(np.int16):         types.int16,
    np.dtype(np.int32):         types.int32,
    np.dtype(np.int64):         types.int64,
    np.dtype(np.uint8):         types.uint8,
    np.dtype(np.uint16): 		types.uint16,
    np.dtype(np.uint32):        types.uint32,
    np.dtype(np.uint64):        types.uint64,
    np.dtype(np.float32):       types.float32,
    np.dtype(np.float64):       types.float64,
    np.dtype(np.complex64):     types.complex64,
    np.dtype(np.complex128):    types.complex128,
}

def rule_numba_type(infer, value, obj):
    value.replace(value=obj.get_dtype())

def rule_shared_array(infer, value):
    nargs = len(value.args.args)
    if nargs != 2:
        msg = "cuda.shared.array() takes eactly two arguments"
        raise CudaPyInferError(value, msg)
    shape_argref, dtype_argref = value.args.args

    shape = extract_shape_arg(shape_argref)
    value.args.args[0] = shape # normalize

    dtype_arg = dtype_argref.value
    if dtype_arg.kind != 'Global':
        msg = "dtype must be a global constant"
        raise CudaPyInferError(dtype_arg, msg)

    dtype = dtype_arg.args.value

    ndim = len(shape)

    elemtype = NP_DTYPE_MAP[dtype]
    arytype = types.arraytype(elemtype, ndim, 'C')

    infer.rules[value].add(MustBe(arytype))
    infer.possible_types.add(arytype)

    value.replace(func=ptx.shared.array)


#-------------------------------------------------------------------------------

def extract_shape_arg(shape_argref):
    if not isinstance(shape_argref, tuple):
        assert shape_argref.value.kind == 'Const'
        if isinstance(shape_argref.value.args.value, tuple):
            shape = shape_argref.value.args.value
            assert all(isinstance(x, int) for x in shape)
            return shape

        shape_argref = (shape_argref,)


    shape_args = tuple(x.value for x in shape_argref)
    for sarg in shape_args:
        if sarg.kind != 'Const':
            msg = "shape must be a constant"
            raise CudaPyInferError(sarg, msg)
    shape = tuple(x.args.value for x in shape_args)
    return shape

#-------------------------------------------------------------------------------

cudapy_global_typing_ext = {
    # indices
    ptx.threadIdx.x: rule_sreg,
    ptx.threadIdx.y: rule_sreg,
    ptx.threadIdx.z: rule_sreg,
    ptx.blockIdx.x:  rule_sreg,
    ptx.blockIdx.y:  rule_sreg,
    # dimensions
    ptx.blockDim.x:  rule_sreg,
    ptx.blockDim.y:  rule_sreg,
    ptx.blockDim.z:  rule_sreg,
    ptx.gridDim.x:   rule_sreg,
    ptx.gridDim.y:   rule_sreg,
}

npy_dtype_ext = {
    np.int8:        rule_np_dtype,
    np.int16:       rule_np_dtype,
    np.int32:       rule_np_dtype,
    np.int64:       rule_np_dtype,
    np.uint8:       rule_np_dtype,
    np.uint16:      rule_np_dtype,
    np.uint32:      rule_np_dtype,
    np.uint64:      rule_np_dtype,
    np.float32:     rule_np_dtype,
    np.float64:     rule_np_dtype,
    np.complex64: 	rule_np_dtype,
    np.complex128:  rule_np_dtype,
}

numba_type_ext = {
    numbapro.int8:          rule_numba_type,
    numbapro.int16:         rule_numba_type,
    numbapro.int32:         rule_numba_type,
    numbapro.int64:         rule_numba_type,
    numbapro.uint8:         rule_numba_type,
    numbapro.uint16:        rule_numba_type,
    numbapro.uint32:        rule_numba_type,
    numbapro.uint64:        rule_numba_type,
    numbapro.float32:       rule_numba_type,
    numbapro.float64:       rule_numba_type,
    numbapro.complex64:     rule_numba_type,
    numbapro.complex128:    rule_numba_type,
}

cudapy_call_typing_ext = {
    # grid:
    ptx.grid:           rule_grid_macro,
    # syncthreads:
    ptx.syncthreads:    rule_syncthread,
    # shared
    ptx.shared.array:   rule_shared_array,
}

cudapy_global_typing_ext.update(npy_dtype_ext)
cudapy_global_typing_ext.update(numba_type_ext)



