import numpy as np
from numbapro.npm.types import *
from numbapro.npm.typing import (cast_penalty, Restrict, UserType, MustBe,
                                 Conditional, int_set, float_set,
                                 filter_array)
from numbapro.npm.errors import CompileError
from . import ptx

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
        tuplety = tupletype(ptx.SREG_TYPE, 2)
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
    np.dtype(np.int8): int8,
    np.dtype(np.int16): int16,
    np.dtype(np.int32): int32,
    np.dtype(np.int64): int64,
    np.dtype(np.uint8): uint8,
    np.dtype(np.uint16): uint16,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
    np.dtype(np.float32): float32,
    np.dtype(np.float64): float64,
    np.dtype(np.complex64): complex64,
    np.dtype(np.complex128): complex128,
}

def rule_shared_array(infer, value):
    args = map(lambda x: x.value, value.args.args)
    if len(args) != 2:
        msg = "cuda.shared.array() takes eactly two arguments"
        raise CudaPyInferError(value, msg)
    shape_arg, dtype_arg = args
    if shape_arg.kind != 'Const':
        msg = "shape must be a constant"
        raise CudaPyInferError(shape_arg, msg)
    if dtype_arg.kind != 'Global':
        msg = "dtype must be a global constant"
        raise CudaPyInferError(dtype_arg, msg)

    shape_value = shape_arg.args.value
    if isinstance(shape_value, tuple):
        shape = shape_value
    else:
        shape = (shape_value,)

    dtype = dtype_arg.args.value

    ndim = len(shape)

    elemtype = NP_DTYPE_MAP[dtype]
    arytype = arraytype(elemtype, ndim, 'C')

    infer.rules[value].add(MustBe(arytype))
    infer.possible_types.add(arytype)

    value.replace(func=ptx.shared.array)

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
    np.int8: rule_np_dtype,
    np.int16: rule_np_dtype,
    np.int32: rule_np_dtype,
    np.int64: rule_np_dtype,
    np.uint8: rule_np_dtype,
    np.uint16: rule_np_dtype,
    np.uint32: rule_np_dtype,
    np.uint64: rule_np_dtype,
    np.float32: rule_np_dtype,
    np.float64: rule_np_dtype,
    np.complex64: rule_np_dtype,
    np.complex128: rule_np_dtype,
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

