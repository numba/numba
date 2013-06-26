from numba import autojit, jit, prange

@autojit
def prange_redux():
    c = 0
    a = 1
    for i in prange(10):
        c += a
    return c

if __name__ == '__main__':
    prange_redux()