import numpy as np
from ..compiler import compile
from ..types import *
from .support import testcase, main

def loop_case_1(a, b):
    '''The type inference algorithm will use the type of at the dominator
    for a variable that participate in a cyclic controlflow structure.
    
    In the following code, `j = 0` will forces `j` to be an int in the for loop.
    But `j *= 2.0` will redefine `j` to be a double in the latter loop.
    '''
    j = 0                       # j is an int
    for i in xrange(a, b):
        k = i * 2.0
        j += k                  # j remains as an int

    j *= 2.0                    # j is a double now
    for i in xrange(a, b):
        k = i * 2.0
        j += k

    return j


def template(func, compiled, args, allclose=False):
    got = compiled(*args)
    exp = func(*args)

    msg = '%s%s got = %s expect=%s' % (func, args, got, exp)

    if allclose:
        assert np.allclose(got, exp), msg
    else:
        assert got == exp, msg


#------------------------------------------------------------------------------
# test_loop_case_1

@testcase
def test_loop_case_1():
    func = loop_case_1
    cfunc = compile(func, float64, [int32, int32])

    args = 0, 100
    template(func, cfunc, args)

if __name__ == '__main__':
    main()
