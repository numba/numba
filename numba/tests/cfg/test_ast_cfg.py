from numba import *

@jit(void(int_)) # directives={'control_flow.dot_output': 'out.dot'})
def func(x):
    i = 0
    while i < 10:
        if x > i:
            print x
            y = 14
        else:
            print y

if __name__ == '__main__':
    pass