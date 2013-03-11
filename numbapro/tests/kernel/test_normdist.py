import numpy as np
from contextlib import closing
from math import pi
import unittest
#import matplotlib.pyplot as plt

from numbapro import CU, uint32, double

A = 1664525
C = 1013904223

p0 = 0.322232431088
q0 = 0.099348462606
p1 = 1.0
q1 = 0.588581570495
p2 = 0.342242088547
q2 = 0.531103462366;
p3 = 0.204231210245e-1
q3 = 0.103537752850
p4 = 0.453642210148e-4
q4 = 0.385607006340e-2

def normal(tid, out, seeds):
    seed = seeds[tid]
    randint = abs(A * seed + C) % 0xfffffff
    seeds[tid] = randint
    u = randint / double(0xfffffff)

    if u < 0.5:
        t = np.sqrt(-2.0 * np.log(u))
    else:
        t = np.sqrt(-2.0 * np.log(1.0 - u))

    p = p0 + t * (p1 + t * (p2 + t * (p3 + t * p4)))
    q = q0 + t * (q1 + t * (q2 + t * (q3 + t * q4)))
    if u < 0.5:
        z = (p / q) - t
    else:
        z = t - (p / q)
    out[tid] = z

class TestNormDist(unittest.TestCase):
    def test_cpu(self):
        self._template('cpu')

    def test_gpu(self):
        self._template('gpu')

    def _template(self, target):
        n = 1000000
        with closing(CU(target)) as cu:
            seed = np.random.random_integers(0, 0xffffffff,
                                             size=n).astype(np.int32)
            normdist = np.empty(n, dtype=np.double)
            d_seed = cu.input(seed)
            d_normdist = cu.output(normdist)
            cu.enqueue(normal, ntid=n, args=(d_normdist, d_seed))
            cu.wait()
        hist, bins = np.histogram(normdist, bins=50, normed=True)
        gold_norm = np.random.randn(n)
        gold_hist, gold_bins = np.histogram(gold_norm, bins=50, normed=True)

        self.assertTrue(np.all(np.abs(bins - gold_bins) < 1))
        err_hist = ((hist - gold_hist) ** 2).sum()
        self.assertTrue(np.all(err_hist < 0.5))
#        plt.hist(normdist, 50, normed=1)
#        plt.hist(gold_norm, 50, normed=1)
#        plt.show()

if __name__ == '__main__':
    unittest.main()
