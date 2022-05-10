from numba import cuda


@cuda.jit("i8(i8)", device=True)
def fib1(n):
    if n < 2:
        return n
    # Note the second call doesn't use a named argument, unlike the cpu usecase
    return fib1(n - 1) + fib1(n - 2)


#def make_fib2():
#    @jit("i8(i8)", nopython=True)
#    def fib2(n):
#        if n < 2:
#            return n
#        return fib2(n - 1) + fib2(n=n - 2)
#
#    return fib2
#
#fib2 = make_fib2()
#
#
#def make_type_change_self(jit=lambda x: x):
#    @jit
#    def type_change_self(x, y):
#        if x > 1 and y > 0:
#            return x + type_change_self(x - y, y)
#        else:
#            return y
#    return type_change_self
#
#
# Implicit signature
@cuda.jit
def fib3(n):
    if n < 2:
        return n

    return fib3(n - 1) + fib3(n - 2)
