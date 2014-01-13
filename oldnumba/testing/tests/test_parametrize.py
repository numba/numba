from numba.testing.test_support import parametrize

@parametrize('foo', 'bar')
def func(arg):
    return arg

assert func_testcase.__name__ == 'func'

assert hasattr(func_testcase, 'test_func_0')
assert hasattr(func_testcase, 'test_func_1')

assert func_testcase('test_func_0').test_func_0() == 'foo'
assert func_testcase('test_func_1').test_func_1() == 'bar'
