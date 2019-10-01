
def ex_literally_usage():
    # magictoken.ex_literally_usage.begin
    import numba

    def power(x, n):
        raise NotImplementedError

    @numba.extending.overload(power)
    def ov_power(x, n):
        if isinstance(n, numba.types.Literal):
            # only if `n` is a literal
            if n.literal_value == 2:
                # special case: square
                print("square")
                return lambda x, n: x * x
            elif n.literal_value == 3:
                # special case: cubic
                print("cubic")
                return lambda x, n: x * x * x

        print("generic")
        return lambda x, n: x ** n

    @numba.njit
    def test_power(x, n):
        return power(x, numba.literally(n))

    # should print "square" and "9"
    print(test_power(3, 2))

    # should print "cubic" and "27"
    print(test_power(3, 3))

    # should print "generic" and "81"
    print(test_power(3, 4))

    # magictoken.ex_literally_usage.end
    assert test_power(3, 2) == 3 ** 2
    assert test_power(3, 3) == 3 ** 3
    assert test_power(3, 4) == 3 ** 4


ex_literally_usage()
