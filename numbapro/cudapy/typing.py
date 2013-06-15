from numbapro.npm.types import *
from numbapro.npm.typing import (cast_penalty, Restrict,
                                 Conditional, int_set, float_set)
from . import ptx

def cast_from_uint32(value):
    return cast_penalty(uint32, value)

def rule_sreg(sreg_stub):
    def _rule(rules, value):
        rules[value].add(Conditional(cast_from_uint32))
        rules[value].add(Restrict(int_set))
        value.replace(value=sreg_stub)
    return _rule

def rule_grid_macro(rules, value):
    rules[value].add(Conditional(cast_from_uint32))
    rules[value].add(Restrict(int_set))
    value.replace(func=ptx.grid)

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
