from __future__ import print_function, absolute_import, division
from numbapro.testsupport import unittest
from numbapro import cuda


def round100(x):
    return round(x * 100)


@cuda.jit("void(float32[:], float32[:])")
def foo(x, y):
    i = cuda.grid(1)
    if i < x.size:
        x[i] += x[i] * y[i]


class TestAutotune(unittest.TestCase):
    def test_autotune_occupancy_exercise(self):
        self.assertTrue(foo.autotune.best() > 0)
        self.assertEqual(foo.autotune.max_occupancy_min_blocks(),
                         foo.autotune.best())
        self.assertTrue(foo.autotune.closest(128) > 0)
        self.assertTrue(foo.autotune.best_within(768, 1024) > 0)

    def test_calc_occupancy(self):
        from numbapro import cuda

        autotuner = cuda.calc_occupancy(cc=(2, 0), reg=32, smem=1200)
        self.assertTrue(autotuner.best() == 128)


if __name__ == '__main__':
    unittest.main()

