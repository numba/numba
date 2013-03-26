from numba.testing.test_support import parametrize

@parametrize('foo', 'bar')
def func(arg):
    return arg

assert func_testcase.__name__ == 'func'

assert hasattr(func_testcase, 'func_0')
assert hasattr(func_testcase, 'func_1')

assert func_testcase('func_0').func_0() == 'foo'
assert func_testcase('func_1').func_1() == 'bar'
