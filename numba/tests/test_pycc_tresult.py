import os
from ctypes import *
from numba import PY3
from numba.pycc import find_shared_ending
from numba.pycc import pycc

base_path = os.path.dirname(os.path.abspath(__file__))

def test_pycc():
    modulename = os.path.join(base_path, 'compile_with_pycc')
    cdll_modulename = modulename + find_shared_ending()
    if os.path.exists(cdll_modulename):
        os.unlink(cdll_modulename)

    pycc.main(args=[modulename + '.py'])
    lib = CDLL(cdll_modulename)

    try:
        lib.mult.argtypes = [c_double, c_double]
        lib.mult.restype = c_double

        lib.multf.argtypes = [c_float, c_float]
        lib.multf.restype = c_float

        res = lib.mult(123, 321)
        print('lib.mult(123, 321) = %f', res)
        assert res == 123 * 321

        res = lib.multf(987, 321)
        print('lib.multf(987, 321) = %f' % res)
        assert res == 987 * 321
    finally:
        del lib
        if os.path.exists(cdll_modulename):
            os.unlink(cdll_modulename)

    out_modulename = (os.path.join(base_path, 'compiled_with_pycc')
                      + find_shared_ending())
    pycc.main(args=['--python', '-o', out_modulename, modulename + '.py'])
    try:
        import numba.tests.compiled_with_pycc as lib
        try:
            res = lib.mult(123, 321)
            print('lib.mult(123, 321) = %f' % res)
            assert res == 123 * 321

            res = lib.multf(987, 321)
            print('lib.multf(987, 321) = %f', res)
            assert res == 987 * 321
        finally:
            del lib
    finally:
        if os.path.exists(out_modulename):
            os.unlink(out_modulename)

if __name__ == "__main__":
    if PY3:
        print('pycc is not yet supported in Python 3')
    else:
        test_pycc()
