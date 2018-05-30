
from ndtypes import ndt

from xnd import xnd

import numpy as np

from numba.gumath import jit_xnd
from numba.gumath.llvm import xnd_t
from numba import jit
from numba.typing.typeof import typeof_impl
from numba.types.abstract import Type
from numba.types.common import SimpleIterableType, SimpleIteratorType
from numba import types
from numba.datamodel import register_default
from numba.datamodel.models import DataModel, StructModel


FROM_SCALAR = {
    ndt('bool'): types.boolean,
    ndt('int8'): types.int8,
    ndt('int16'): types.int16,
    ndt('int32'): types.int32,
    ndt('int64'): types.int64,

    ndt('uint8'): types.uint8,
    ndt('uint16'): types.uint16,
    ndt('uint32'): types.uint32,
    ndt('uint64'): types.uint64,

    ndt('float32'): types.float32,
    ndt('float64'): types.float64,

    ndt('complex64'): types.complex64,
    ndt('complex128'): types.complex128,
}

def type_from_scalar(x: xnd):
    return FROM_SCALAR[x.type.hidden_dtype]

class XNDIterator(SimpleIteratorType):
    def __init__(self, xnd_type): 
        self.xnd_type = xnd_type
        x = xnd_type.xnd
        super(XNDIterator, self).__init__(
            f'iter(xnd({x.type}))',
            XNDType(x[0]) if x.ndim > 1 else type_from_scalar(x)
        )


class XNDType(SimpleIterableType):
    def __init__(self, x: xnd):
        self.xnd = x
        super(XNDType, self).__init__(
            f'xnd({x.type})',
            XNDIterator(self)
        )

@typeof_impl.register(xnd)
def typeof_xnd(val, c):
    if val.ndim == 0:
        raise NotImplementedError('Scalar xnd inputs aren\'t implemented')
    return XNDType(val)

@register_default(XNDIterator)
class XNDIteratorModel(StructModel):
    def __init__(self, dmm, fe_type: XNDIterator):
        # We use an unsigned index to avoid the cost of negative index tests.
        members = [('index', types.EphemeralPointer(types.uintp)),
                   ('array', fe_type.xnd_type)]
        super(XNDIteratorModel, self).__init__(dmm, fe_type, members)



@register_default(XNDType)
class XNDModel(DataModel):
    def get_value_type(self):
        return xnd_t
    
    def from_argument(self, builder, val):
        return val


@jit(nopython=True)
def do_things(x: xnd) -> int:
    y = 0
    for i in x:
        for j in i:
            y += j
    return y


# TODO, add @lower_builtin('getiter', XNDType)
# maybe change xnd data model to be accurate to python data model? 
# Is it possible to get xnd pointer from python xnd?


# do_things(np.array(10))
do_things(xnd([[1, 2, 3], [4, 5, 6]]))

# @vectorize
# def f(a):
#     return a

# @vectorize
# def g(b):
#     return f(b)

# print(g(np.arange(10)))


# print(f(np.arange(10)))

# @jit_xnd
# def eql(a, b):
#     return a  == b
# eql(xnd(1), xnd(1))

# @jit_xnd
# def add_two(a, b):
#     return a + b

# print(add_two(xnd([2, 1, 1]), xnd([4, 5, 6])))

# print(add_two(xnd([2.3, 1.3, 1.3]), xnd([4.33, 5.3, 6.3])))

# one = xnd.from_buffer(np.arange(100).astype(np.float32))
# print(add_two(one, one))


# @jit_xnd('... * float64 -> ... * float64')
# def sin_thing(a):
#     return np.sin(a)

# print(sin_thing(xnd([[2.0], [1.0], [23.0]])))


# @jit_xnd('K * N * int64, int64 -> K * N * int64')
# def g(x, y, res):
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             res[i][j] = x[i][j] + y


# print(g(xnd([[2, 3, 4], [5, 6, 7]]), xnd(4)))


# # # using notation from https://en.wikipedia.org/wiki/Matrix_multiplication#Definition
# @jit_xnd([
#     'N * M * float64, M * P * float64 -> N * P * float64',
#     'N * M * int64, M * P * int64 -> N * P * int64',
# ])
# def matrix_multiply(a, b, c):
#     n, m = a.shape
#     m, p = b.shape
#     for i in range(n):
#         for j in range(p):
#             c[i][j] = 0
#             for k in range(m):
#                 c[i][j] += a[i][k] * b[k][j]


# print(matrix_multiply(
#     xnd([[0, 1], [0, 0]]),
#     xnd([[0, 0], [1, 0]]) 
# ))

# print(matrix_multiply(
#     xnd([[-2, 5], [1, 6], [-4, -1]]),
#     xnd([[2, 7], [8, -3]]) 
# ))


# print(matrix_multiply(
#     xnd([[123.023]]),
#     xnd([[23.2323]]) 
# ))
