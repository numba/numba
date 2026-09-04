from numba import njit


@njit
def foo(x):
    return x + 1


res = foo(123)
assert res == 123 + 1
print(res)