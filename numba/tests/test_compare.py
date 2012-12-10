
from numba import autojit

def _make_test(f):
    def test():
        f_ = autojit(f)
        for v in range(-10,10):
            assert f_(v)==f(v)
            assert f_(float(v))==f(float(v))
    test.func_name = f.func_name
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
