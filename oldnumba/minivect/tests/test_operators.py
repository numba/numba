# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from .llvm_testutils import *

def build_expr(type, op):
    out, v1, v2 = vars = build_vars(type, type, type)
    expr = b.assign(out, b.binop(type, op, v1, v2))
    return vars, expr

def build_kernel(specialization_name, ndim, type, op, **kw):
    vars, expr = build_expr(minitypes.ArrayType(type, ndim, **kw), op)
    func = MiniFunction(specialization_name, vars, expr, '%s_%s_%s' % (specialization_name, type.name, op))
    return func

comparison_operators = ['<', '<=', '>', '>=', '==', '!=']
arithmetic_operators = ['+', '-', '*', '/', '%'] # + ['**'] + # + comparison_operators
#bitwise_operators = ['<<', '>>', '|', '^', '&']
bitwise_operators = ['|', '^', '&']

a = np.random.random_sample((10, 20))

def _impl(type, op, x, y):
    func = build_kernel('strided', 2, type, op)

    dtype = minitypes.map_minitype_to_dtype(type)
    x = x.astype(dtype)
    y = y.astype(dtype)

    numpy_result = eval('a %s b' % (op,), {'a': x, 'b': y})
    our_result = func(x, y)
    assert np.all(numpy_result == our_result)

@parametrize(type=[short, int32, int64, float_, double], op=arithmetic_operators)
def test_arithmetic_operators(type, op):
    x = a
    y = np.arange(1, 10 * 20 + 1).reshape(10, 20)
    _impl(type, op, x, y)

@parametrize(type=[short, int32, int64], op=bitwise_operators)
def test_bitwise_operators(type, op):
    _impl(type, op, a * 100, a * 10)

