from numbapro import cuda
from numbapro.npm.types import *
from numbapro.npm.typing import (cast_penalty, Restrict,
                                              Conditional, int_set, float_set)


def rule_threadIdx_x(rules, value):
    def cast_from_uint32(typemap, value):
        return cast_penalty(uint32, typemap[value])
    rules[value].add(Conditional(cast_from_uint32))
    rules[value].add(Restrict(int_set))
    value.replace(value=cuda.threadIdx.x)

#-------------------------------------------------------------------------------

cudapy_typing_ext = {
    cuda.threadIdx.x: rule_threadIdx_x,
}