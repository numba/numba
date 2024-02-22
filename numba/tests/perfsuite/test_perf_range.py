import unittest

import numpy as np

from numba.tests.perfsuite_support import PerfTestCase
from numba import njit


class TestFindChrNextOccurance(PerfTestCase):
    """
    The benchmark code is inspired and adapted from:
    https://www.modular.com/blog/outperforming-rust-benchmarks-with-mojo
    """
    def test_find_chr_next_occurance(self):
        @njit
        def find_chr_next_occurance(in_tensor, chr=10, start=0):
            for i in range(start, in_tensor.size):
                if in_tensor[i] == chr:
                    return i
            return -1

        @njit
        def determine_output(temp, start):
            for j in range(temp.size):
                if temp[j]:
                    return start + j
            return -1

        @njit
        def find_chr_next_occurance_simd(in_tensor, chr=10, start=0):
            simd_size = 32
            temp = np.zeros(simd_size, dtype=in_tensor.dtype)

            for chkid in range(start, in_tensor.size, simd_size):
                for j in range(simd_size):
                    i = chkid + j
                    temp[j] = in_tensor[i] == chr

                out = determine_output(temp, chkid)
                if out == -1:
                    temp.fill(0)
                else:
                    return out

                chkid += simd_size

            # peel
            for i in range(chkid, in_tensor.size):
                temp[j] = in_tensor[i] == chr
            return determine_output(temp, chkid)

        # warm up JIT and check correctness
        np.random.seed(0)
        looking_for = 10
        data = np.random.randint(0, 9 + 1, size=10).astype(np.int8)
        data[-1] = looking_for

        result = find_chr_next_occurance_simd(data)
        expect = find_chr_next_occurance.py_func(data)
        self.assertEqual(result, expect)

        data[data.size // 2] = looking_for
        result = find_chr_next_occurance_simd(data)
        expect = find_chr_next_occurance.py_func(data)
        self.assertEqual(result, expect)

        # benchmark execution time
        looking_for = 10
        data = np.random.randint(0, 9 + 1, size=1_000_000).astype(np.int8)
        data[-1] = looking_for

        @self.benchmark
        def execution_time():
            find_chr_next_occurance_simd(data, looking_for)

        # benchmark compile time
        py_func = find_chr_next_occurance_simd.py_func
        sig = find_chr_next_occurance_simd.nopython_signatures[0]

        @self.benchmark
        def compile_time():
            njit(py_func).compile(sig)


if __name__ == '__main__':
    unittest.main()
