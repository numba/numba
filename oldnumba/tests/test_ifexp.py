from itertools import product
from numba import autojit

def _make_test(f):
    def test_ifexp():
        f_ = autojit(f)
        for args in product(range(3), range(3)):
            assert f_(*args)==f(*args)
    test_ifexp.__name__ = f.__name__
    return test_ifexp

@_make_test
def test_as_return_value(a,b):
    return a if a>b else b

@_make_test
def test_assign_and_return(a,b):
    c = a if a>b else b
    return c

@_make_test
def test_in_expression(a,b):
    c = 5 + (a if a>b else b)/2.0
    return c

@_make_test
def test_expr_as_then_clause(a,b):
    return (a+1) if a>b else b

@_make_test
def test_expr_as_else_clause(a,b):
    return a if a>b else (b+1)


@autojit
def _f1(a,b):
    return a if a>b else b

def test_type_promotion():
    assert isinstance(_f1(1, 1), (int, long))
    assert isinstance(_f1(1.0, 1), float)
    assert isinstance(_f1(1, 1.0), float)

if __name__ == '__main__':
    test_type_promotion()
