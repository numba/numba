from numba import njit
import numpy as np


def wrapper(exc):
    @njit('string(int64, string)', no_cfunc_wrapper=True)
    def foo(i, a):
        if i == 0:
            raise IndexError(a + ' world', a, 'test', a, 3)
        elif i == 1:
            raise ValueError(a)
        elif i == 2:
            raise exc(a)
        else:
            return a

    return foo

@njit(no_cfunc_wrapper=True)
def bar():
    raise ValueError('test', 'hello world', foo, bar, IndexError)


class MyException(Exception):
    pass


foo = wrapper(MyException)
with open('a.ll', 'w') as f:
    print(foo.inspect_llvm(foo.signatures[0]), file=f)

try:
    print(foo(2, 'hello'))
except Exception as e:
    print('Exception: ', e, type(e))
    pass

# try:
#     bar()
# except Exception as e:
#     print(e)
