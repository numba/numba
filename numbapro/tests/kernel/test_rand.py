import numpy as np
from numbapro.parallel.kernel import CU
from numbapro.parallel.kernel import builtins
from numbapro.cudalib import curand
from numbapro import npy_intp

def randgather(tid, rnd, vals, out):
    idx = npy_intp(rnd[tid] * out.shape[0]) % out.shape[0]
    out[tid] = vals[idx]

def test():
    n = 100
    vals = np.arange(n)
    print vals.dtype, vals

    out = np.empty_like(vals)

    cu = CU('gpu')

    drng = cu.scratch(n, dtype=np.float32)
    dvals = cu.input(vals)
    dout = cu.output(out)

    cu.enqueue(builtins.random.uniform,
               ntid=drng.size, args=(drng,))
    
    cu.enqueue(randgather,
               ntid=dout.size, args=(drng, dvals, dout))

    cu.wait()

    print dout.size, dout



if __name__ == '__main__':
    test()
