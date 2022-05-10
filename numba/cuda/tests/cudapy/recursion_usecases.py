from numba import cuda


@cuda.jit("i8(i8)", device=True)
def fib1(n):
    if n < 2:
        return n
    # Note the second call uses a named argument
    return fib1(n - 1) + fib1(n=n - 2)


def make_fib2():
    @cuda.jit("i8(i8)", device=True)
    def fib2(n):
        if n < 2:
            return n
        return fib2(n - 1) + fib2(n=n - 2)

    return fib2


fib2 = make_fib2()


@cuda.jit
def type_change_self(x, y):
    if x > 1 and y > 0:
        return x + type_change_self(x - y, y)
    else:
        return y


# Implicit signature
@cuda.jit
def fib3(n):
    if n < 2:
        return n

    return fib3(n - 1) + fib3(n - 2)
