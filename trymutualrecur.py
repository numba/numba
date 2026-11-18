from numba import njit

@njit(cache=True)
def foo(x):
    if x > 0:
        return 2 * bar(z=1, y=x)
    return 1 + x

@njit
def bar(y, z):
    return foo(x=y - z)


res = foo(5)
assert res == foo.py_func(5)
print(res)

