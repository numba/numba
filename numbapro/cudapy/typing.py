from numbapro.npm.types import *
from numbapro.npm.typing import (cast_penalty, Restrict,
                                 Conditional, int_set, float_set)
from numbapro.npm.errors import CompileError
from . import ptx

class CudaPyInferError(CompileError):
    def __init__(self, value, msg):
        super(CudaPyInferError, self).__init__(value.lineno, msg)

def cast_from_sregtype(value):
    return cast_penalty(ptx.SREG_TYPE, value)

def rule_sreg(sreg_stub):
    def _rule(rules, value):
        rules[value].add(Conditional(cast_from_sregtype))
        rules[value].add(Restrict(int_set))
        value.replace(value=sreg_stub)
    return _rule

def rule_grid_macro(rules, value):
    args = value.args.args
    if len(args) != 1:
        raise CudaPyInferError(value, "grid() takes exactly 1 argument")
    arg = args[0].value
    if arg.kind != 'Const':
        raise CudaPyInferError(value, "arg to grid() must be a constant")
    ndim = arg.args.value
    if ndim not in [1, 2]:
        raise CudaPyInferError(value, "arg to grid() must be 1 or 2")

    value.replace(func=ptx.grid)
    if ndim == 1:
        rules[value].add(Conditional(cast_from_sregtype))
        rules[value].add(Restrict(int_set))
        return
    else:
        assert ndim == 2
        tuplety = tupletype(ptx.SREG_TYPE, 2)
        def ret_tuple(value):
             return value == tuplety
        rules[value].add(Conditional(ret_tuple))
        return [tuplety]
#-------------------------------------------------------------------------------

cudapy_global_typing_ext = {
    # indices
    ptx.threadIdx.x: rule_sreg(ptx.threadIdx.x),
    ptx.threadIdx.y: rule_sreg(ptx.threadIdx.y),
    ptx.threadIdx.z: rule_sreg(ptx.threadIdx.z),
    ptx.blockIdx.x:  rule_sreg(ptx.blockIdx.x),
    ptx.blockIdx.y:  rule_sreg(ptx.blockIdx.y),
    # dimensions
    ptx.blockDim.x:  rule_sreg(ptx.blockDim.x),
    ptx.blockDim.y:  rule_sreg(ptx.blockDim.y),
    ptx.blockDim.z:  rule_sreg(ptx.blockDim.z),
    ptx.gridDim.x:   rule_sreg(ptx.gridDim.x),
    ptx.gridDim.y:   rule_sreg(ptx.gridDim.y),
}

cudapy_call_typing_ext = {
    # grid:
    ptx.grid:        rule_grid_macro,
}
