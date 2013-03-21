import numba
from numba import *
from numba.typesystem import typeset
from numba.environment import NumbaEnvironment

def s(type):
    return type(type, type)

def test_typeset_matching():
    context = NumbaEnvironment.get_environment().context

    numeric = typeset.typeset([int_, longlong])
    n = numeric(numeric, numeric)
    f = numba.floating(numba.floating, numba.floating)

    signatures = [n, f, object_(object_, object_)]
    ts = typeset.typeset(signatures)

    assert ts.find_match(context, [float_, float_]) == s(float_)
    assert ts.find_match(context, [float_, double]) == s(double)
    assert ts.find_match(context, [longdouble, float_]) == s(longdouble)

    assert ts.find_match(context, [int_, int_]) == s(int_)
    assert ts.find_match(context, [int_, longlong]) == s(longlong)
    assert ts.find_match(context, [short, int_]) == s(int_)

    assert ts.find_match(context, [short, ulonglong]) is None

if __name__ == '__main__':
    test_typeset_matching()
