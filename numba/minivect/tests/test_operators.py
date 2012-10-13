import pytest

from llvm_testutils import *

def build_expr(type, op):
    out, v1, v2 = vars = build_vars(type, type, type)
    expr = b.assign(out, b.binop(type, op, v1, v2))
    return vars, expr

def build_kernel(specialization_name, ndim, type, op, **kw):
    vars, expr = build_expr(minitypes.ArrayType(type, ndim, **kw), op)
    func = MiniFunction(specialization_name, vars, expr)
    return func

comparison_operators = ['<', '<=', '>', '>=', '==', '!=']
arithmetic_operators = ['+', '-', '*', '/', '%'] # + ['**'] + # + comparison_operators
bitwise_operators = ['<<', '>>', '|', '^', '&']

a = np.random.random_sample((10, 20))

def pytest_generate_tests(metafunc):
    """
    Generate tests for binary operators
    """
    if metafunc.function is test_arithmetic_operators:
        metafunc.parametrize("type", [short, int32, int64, float_, double])
        metafunc.parametrize("op", arithmetic_operators)
    elif metafunc.function is test_bitwise_operators:
        metafunc.parametrize("type", [short, int32, int64])
        metafunc.parametrize("op", bitwise_operators)

def _impl(type, op, x, y):
    func = build_kernel('strided', 2, type, op)

    dtype = minitypes.map_minitype_to_dtype(type)
    x = x.astype(dtype)
    y = y.astype(dtype)

    numpy_result = eval('a %s b' % (op,), {'a': x, 'b': y})
    our_result = func(x, y)
    assert np.all(numpy_result == our_result)

def test_arithmetic_operators(type, op):
    x = a
    y = np.arange(1, 10 * 20 + 1).reshape(10, 20)
    _impl(type, op, x, y)

def test_bitwise_operators(type, op):
    _impl(type, op, a * 100, a * 10)