# Contents in this file are referenced from the sphinx-generated docs.
# "magictoken" is used for markers as beginning and ending of example text.

import unittest
from numba.tests.support import captured_stdout


class ChunksizeExamplesTest(unittest.TestCase):

    def test_chunksize_manual(self):
        with captured_stdout():
            # magictoken.ex_chunksize_manual.begin
            from numba import njit, prange, set_parallel_chunksize, get_parallel_chunksize

            @njit(parallel=True)
            def func1(n):
                acc = 0
                for i in prange(n):
                    acc += i
                return acc

            @njit(parallel=True)
            def func2(n):
                acc = 0
                # This version gets the previous chunksize explicitly.
                old_chunksize = get_parallel_chunksize()
                set_parallel_chunksize(8)
                for i in prange(n):
                    acc += i
                set_parallel_chunksize(old_chunksize)
                return acc

            # This version saves the previous chunksize as returned
            # by set_parallel_chnksize.
            old_chunksize = set_parallel_chunksize(4)
            func1(12)
            func2(12)
            func1(12)
            set_parallel_chunksize(old_chunksize)
            # magictoken.ex_chunksize_manual.end

    def test_chunksize_with(self):
        with captured_stdout():
            # magictoken.ex_chunksize_with.begin
            from numba import njit, prange, parallel_chunksize

            @njit(parallel=True)
            def func1(n):
                acc = 0
                for i in prange(n):
                    acc += i
                return acc

            @njit(parallel=True)
            def func2(n):
                acc = 0
                with parallel_chunksize(8):
                    for i in prange(n):
                        acc += i
                return acc # Note: this cannot go inside the with

            with parallel_chunksize(4):
                func1(12)
                func2(12)
                func1(12)
            # magictoken.ex_chunksize_with.end


if __name__ == '__main__':
    unittest.main()
