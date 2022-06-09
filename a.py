from numba import njit
import numpy as np


def call(fn, *args):
    try:
        out = fn(*args)
    except Exception as e:
        print(f"Exception ({type(e)}): {e}")
    else:
        print(f"result: {out}")


def dump_ir(fn, filename='a.ll'):
    with open(filename, 'w+') as f:
        print(fn.inspect_llvm(fn.signatures[0]), file=f, flush=True)


class MyException(Exception):
    pass


@njit(no_cfunc_wrapper=True)
def bar():
    raise ValueError('test', 'hello world', foo, bar, IndexError)


@njit('string(int64, string)', no_cfunc_wrapper=True)
def foo(i, a):
    if i == 0:
        raise IndexError(a + ' world', a, 'test', a, 3)
    elif i == 1:
        raise ValueError(a, ValueError)
    elif i == 2:
        raise MyException(a, MyException)
    else:
        return a


@njit('string(string)', no_cfunc_wrapper=True)
def baz(a):
    raise ValueError(a, baz)


call(bar)
call(baz, "hello")
for i in range(3):
    call(foo, i, 'hello')

try:
    bar()
except Exception as e:
    print(e)
