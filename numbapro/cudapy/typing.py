import numpy as np
from numbapro.npm import types
from numbapro.npm.typing import (cast_penalty, Restrict, MustBe, filter_array,
                                 Conditional, int_set)
from . import ptx, libdevice
import numbapro

def cast_from_sregtype(value):
    return cast_penalty(ptx.SREG_TYPE, value)

def rule_sreg(infer, value, obj):
    infer.rules[value].add(MustBe(types.uint32))
    value.replace(value=obj)

def rule_grid_macro(infer, value):
    args = value.args.args

    if len(args) != 1:
        raise TypeError("grid() takes exactly 1 argument")
    arg = args[0].value

    if arg.kind != 'Const':
        raise TypeError("arg to grid() must be a constant")
    ndim = arg.args.value

    if ndim not in [1, 2]:
        raise ValueError("arg to grid() must be 1 or 2")

    if ndim == 1:
        infer.rules[value].add(Conditional(cast_from_sregtype))
        infer.rules[value].add(Restrict(int_set))
    else:
        tuplety = types.tupletype(ptx.SREG_TYPE, 2)
        infer.rules[value].add(MustBe(tuplety))
        infer.possible_types.add(tuplety)

    value.replace(func=ptx.grid)

def rule_syncthread(infer, value):
    if value.args.args:
        raise TypeError("syncthreads() takes no arguments")
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

def rule_numba_cast(obj, destty):
    def _cast(infer, value):
        infer.rules[value].add(MustBe(destty))
        value.replace(func=obj)
    return _cast

def rule_shared_array(infer, value):
    # convert kws to args
    kws = dict(value.args.kws)
    argorder = 'shape', 'dtype'

    argvals = list(value.args.args)
    for arg in argorder[len(argvals):]:
        argvals.append(kws.pop(arg))

    if kws:
        raise NameError("duplicated keywords %s" % ','.join(kws.keys()))

    # normalize the call so there will be no kws
    value.replace(args=list(argvals), kws=())

    # add typing rules
    nargs = len(value.args.args)
    if nargs != 2:
        msg = "cuda.shared.array() takes eactly two arguments"
        raise TypeError(msg)
    shape_argref, dtype_argref = value.args.args

    shape = extract_shape_arg(shape_argref)
    value.args.args[0] = shape # normalize

    dtype_arg = dtype_argref.value
    if dtype_arg.kind != 'Global':
        msg = "dtype must be a global constant"
        raise ValueError(msg)

    dtype = dtype_arg.args.value

    ndim = len(shape)

    elemtype = NP_DTYPE_MAP[dtype]
    arytype = types.arraytype(elemtype, ndim, 'C')

    infer.rules[value].add(MustBe(arytype))
    infer.possible_types.add(arytype)

    value.replace(func=ptx.shared.array)

def rule_atomic_add(infer, value):
    assert not value.args.kws, "does not support keyword args"
    argvals = list(value.args.args)
    if len(argvals) != 3:
        raise TypeError("atomic.add takes exactly 3 args")
    arg_ary, arg_idx, ary_val = argvals
    if not isinstance(arg_idx, tuple):
        arg_idx = (arg_idx,)
    getvalue = lambda x: x.value

    # normalize the call args
    value.replace(args=[arg_ary, arg_idx, ary_val], kws=())

    ary = getvalue(arg_ary)
    idx = map(getvalue, arg_idx)
    val = getvalue(ary_val)

    infer.rules[ary].add(Restrict(filter_array(infer.possible_types)))

    def is_element_type(val, ary):
        if ary.is_array:
            return val == ary.element
    infer.rules[value].add(Conditional(is_element_type, ary))

    def match_element_type(val, ary):
        if ary.is_array:
            return cast_penalty(val, ary.element)
    infer.rules[val].add(Conditional(match_element_type, ary))


    def prefer_intp(idx):
        return cast_penalty(idx, infer.intp)
    for i in idx:
        infer.rules[val].add(Restrict(int_set))
        infer.rules[val].add(Conditional(prefer_intp))
    value.replace(func=ptx.atomic.add)

#-------------------------------------------------------------------------------

def extract_shape_arg(shape_argref):
    if not isinstance(shape_argref, tuple):
        assert shape_argref.value.kind == 'Const', "shape is not a constant"
        if isinstance(shape_argref.value.args.value, tuple):
            shape = shape_argref.value.args.value
            assert all(isinstance(x, int) for x in shape), \
                                    "shape must be a constant tuple of ints"
            return shape

        shape_argref = (shape_argref,)


    shape_args = tuple(x.value for x in shape_argref)
    for sarg in shape_args:
        if sarg.kind != 'Const':
            msg = "shape must be a constant"
            raise TypeError(msg)
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

numba_cast_ext = {
    numbapro.int8:          rule_numba_cast(numbapro.int8, types.int8),
    numbapro.int16:         rule_numba_cast(numbapro.int16, types.int16),
    numbapro.int32:         rule_numba_cast(numbapro.int32, types.int32),
    numbapro.int64:         rule_numba_cast(numbapro.int64, types.int64),
    numbapro.uint8:         rule_numba_cast(numbapro.uint8, types.uint8),
    numbapro.uint16:        rule_numba_cast(numbapro.uint16, types.uint16),
    numbapro.uint32:        rule_numba_cast(numbapro.uint32, types.uint32),
    numbapro.uint64:        rule_numba_cast(numbapro.uint64, types.uint64),
    numbapro.float32:       rule_numba_cast(numbapro.float32, types.float32),
    numbapro.float64:       rule_numba_cast(numbapro.float64, types.float64),
    numbapro.complex64:     rule_numba_cast(numbapro.complex64, types.complex64),
    numbapro.complex128:    rule_numba_cast(numbapro.complex128, types.complex128),
}

cudapy_call_typing_ext = {
    # grid:
    ptx.grid:           rule_grid_macro,
    # syncthreads:
    ptx.syncthreads:    rule_syncthread,
    # shared
    ptx.shared.array:   rule_shared_array,
    # atomic
    ptx.atomic.add:     rule_atomic_add,
}

cudapy_call_typing_ext.update(numba_cast_ext)
cudapy_call_typing_ext.update(libdevice.math_infer_rules)

cudapy_global_typing_ext.update(npy_dtype_ext)
cudapy_global_typing_ext.update(numba_type_ext)



