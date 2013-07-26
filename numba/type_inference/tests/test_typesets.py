import numba
from numba import *
from numba.typesystem import typeset, promote
from numba.environment import NumbaEnvironment

def s(type):
    return type(type, type)

def typeset_matching():
    numeric = typeset.typeset([int_, longlong])
    n = numeric(numeric, numeric)
    f = numba.floating(numba.floating, numba.floating)

    signatures = [n, f, object_(object_, object_)]
    ts = typeset.typeset(signatures)

    assert ts.find_match(promote, [float_, float_]) == s(float_)
    assert ts.find_match(promote, [float_, double]) == s(double)
    # assert ts.find_match(promote, [longdouble, float_]) == s(longdouble)

    assert ts.find_match(promote, [int_, int_]) == s(int_)
    # assert ts.find_match(promote, [int_, longlong]) == s(longlong)
    # assert ts.find_match(promote, [short, int_]) == s(int_)

    # np.result_type(np.short, np.ulonglong) -> np.float64
    # assert ts.find_match(promote, [short, ulonglong]) is None

if __name__ == '__main__':
    typeset_matching()
