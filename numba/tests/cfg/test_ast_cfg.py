from numba import *

#@jit(void(int_)) # directives={'control_flow.dot_output': 'out.dot'})
#@jit(void, [int_], backend='bytecode')
@jit(void(int_))
def func(x):
    i = 0
    #y = 12
    while i < 10:
        if x > i:
            print x
            y = 14
        else:
            print y

        i = i + 1

if __name__ == '__main__':
    pass