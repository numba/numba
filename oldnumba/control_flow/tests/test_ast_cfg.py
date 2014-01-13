import numba
from numba import *

if not numba.PY3:
    #@jit(void(int_)) # directives={'control_flow.dot_output': 'out.dot'})
    #@jit(void, [int_], backend='bytecode')
    @jit(void(int_))
    def func(x):
        i = 0
        #y = 12
        h = 30
        print(i)
        while i < 10:
            if x > i:
                print(x)
                y = 14
            else:
                print(y)

            i = i + 1
            print(y)

        print(i)
        print(y)

#@jit(void())
def _for_loop_fn_0():
    acc = 0.
    for value in range(10):
        acc += value
    return acc

#@jit(void(int_, float_))
def func(a, b):
    if a:
        c = 2
    else:
        c = double(4)

    if a:
        c = 4
    #while a < 4:
    #    for i in range(10):
    #        b = 9
    print(b)

if __name__ == '__main__':
    pass
