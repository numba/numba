from numba import njit
import numpy as np


@njit('void(string)', no_cfunc_wrapper=True)
def foo(a):
    raise IndexError(a + ' world', a)

@njit(no_cfunc_wrapper=True)
def bar():
    raise ValueError('teste', 'hello world')


with open('a.ll', 'w') as f:
    print(foo.inspect_llvm(foo.signatures[0]), file=f)

try:
    foo('hello')
except Exception as e:
    print('Exception: ', e)
    pass

try:
    bar()
except Exception as e:
    print(e)
