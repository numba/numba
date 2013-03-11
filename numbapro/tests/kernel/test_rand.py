from __future__ import print_function
import unittest
from contextlib import closing
import numpy as np
from numbapro.parallel.kernel import CU
from numbapro.parallel.kernel import builtins
from numbapro import npy_intp

def randgather(tid, rnd, vals, out):
    idx = npy_intp(rnd[tid] * out.shape[0]) % out.shape[0]
    out[tid] = vals[idx]

class TestRand(unittest.TestCase):
    def test_gpu(self):
        with closing(CU('gpu')) as cu:
            self._template(cu)

# TODO
#    def test_cpu(self):
#        with closing(CU('cpu')) as cu:
#            self._template(cu)

    def _template(self, cu):
        n = 100
        vals = np.arange(n)
#        print(vals.dtype, vals)

        out = np.empty_like(vals)

        drng  = cu.scratch(n, dtype=np.float32)
        dvals = cu.input(vals)
        dout  = cu.output(out)

        cu.configure(builtins.random.seed, 0xbeef)

        cu.enqueue(builtins.random.uniform,
                   ntid=drng.size, args=(drng,))

        cu.enqueue(randgather,
                   ntid=dout.size, args=(drng, dvals, dout))

        cu.wait()

#        print(out.size, out)

        # check
        self.assertTrue(np.all(out < out.size))
        self.assertTrue(np.all(out >= 0))
        diff = (out != vals).sum()
        self.assertTrue(diff > (out.size // 2),
                        "Expect more than 50% of values be different")


if __name__ == '__main__':
    unittest.main()
