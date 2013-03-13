from numba.tests.test_support import parametrize

@parametrize('foo', 'bar')
def func(arg):
    return arg

assert func_0() == 'foo'
assert func_1() == 'bar'