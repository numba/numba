
from numba import autojit

tests = []

def _make_test(f):
    def test():
        f_ = autojit(f)
        for v in range(-10,10):
            assert f_(v)==f(v)
            assert f_(float(v))==f(float(v))

    test.func_name = f.func_name
    tests.append(test)
    return test

@_make_test
def test_single_comparator(a):
    return a<4

@_make_test
def test_single_float_comparator(a):
    return a<4.0

@_make_test
def test_multiple_comparators(a):
    return 0<a<=4

@_make_test
def test_multiple_comparators_mixed_types(a):
    return 0.0<a<=10

@_make_test
def test_compare_span_basic_blocks(a):
    a = a + 1j
    if abs(a) > 2:
        return 0 < abs(a) < 10

    return not a.real > 0

@_make_test
def test_compare_while(a):
    while 1:
        while 1:
            break
        else:
            print "hello"
            return a * 3
        break
    return a * 2

if __name__ == "__main__":
    # autojit(test_compare_span_basic_blocks)(5)
#    autojit(test_compare_while)(10)
    for test in tests:
        test()
#    import numba
#    numba.nose_run()