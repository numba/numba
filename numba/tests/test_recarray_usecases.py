import numpy
from numba import numpy_support
from numba.compiler import compile_isolated
from numba import unittest_support as unittest

mystruct_dt = numpy.dtype([('p', numpy.float64),
                           ('row', numpy.float64),
                           ('col', numpy.float64)])

mystruct = numpy_support.from_dtype(mystruct_dt)


def usecase1(arr1, arr2):
    """Base on https://github.com/numba/numba/issues/370

    Modified to add test-able side effect.
    """
    n1 = arr1.size
    n2 = arr2.size

    for i1 in range(n1):
        st1 = arr1[i1]
        for i2 in range(n2):
            st2 = arr2[i2]
            st2.row += st1.p * st2.p + st1.row - st1.col

        st1.p += st2.p
        st1.col -= st2.col


class TestRecordUsecase(unittest.TestCase):
    def test_usecase1(self):
        pyfunc = usecase1
        cres = compile_isolated(pyfunc, (mystruct[:], mystruct[:]))
        cfunc = cres.entry_point

        st1 = numpy.recarray(3, dtype=mystruct_dt)
        st2 = numpy.recarray(3, dtype=mystruct_dt)

        st1.p = numpy.arange(st1.size) + 1
        st1.row = numpy.arange(st1.size) + 1
        st1.col = numpy.arange(st1.size) + 1

        st2.p = numpy.arange(st2.size) + 1
        st2.row = numpy.arange(st2.size) + 1
        st2.col = numpy.arange(st2.size) + 1

        expect1 = st1.copy()
        expect2 = st2.copy()

        got1 = expect1.copy()
        got2 = expect2.copy()

        pyfunc(expect1, expect2)
        cfunc(got1, got2)

        self.assertTrue(numpy.all(expect1 == got1))
        self.assertTrue(numpy.all(expect2 == got2))


if __name__ == '__main__':
    unittest.main()
