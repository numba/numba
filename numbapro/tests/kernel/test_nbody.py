"""
Test infinite loop for GPU CU auto block size selection
Submitted by Christopher Cooper
"""

## Nbody with Numbapro.
## This Nbody emulates the P2P in a Treecode, as not all
## particles interact with each other.
## Particles are divided into 5 chunks and each chunk interacts
## with particles of its own chunk and neighbouring chunks (ie.
## chunck 0 interacts with 0 and 1, chunk 1 with 0, 1 and 2, etc.)

from contextlib import closing
import numpy as np
from numbapro import CU, cuda
import math
import time
import unittest

def nbody(tid, xi, yi, zi, xj, yj, zj, m, result,
         chunk_size, offset_particles, inter_list, offset_chunks, sort_particles, eps):
    chunkId = tid/chunk_size[0]
    CJ_start = offset_chunks[chunkId]
    CJ_end   = offset_chunks[chunkId+1]
    result[tid] = 0. 
    for CJ in inter_list[CJ_start:CJ_end]:
        for j in sort_particles[offset_particles[CJ]:offset_particles[CJ+1]]:
            dx = xi[tid]-xj[j]
            dy = yi[tid]-yj[j]
            dz = zi[tid]-zj[j]
#            result[tid] += m[j]/np.sqrt(dx*dx+dy*dy+dz*dz+eps[0])
            result[tid] += m[j]/np.sqrt(dx*dx+dy*dy+dz*dz+1e-10)

# The code hangs if:
# 	- I replace 1e-10 by eps[0] (value that comes from the CPU) 
# 	- I use a cached register to accumulate the sum instead of storing in result[tid] in every iteration
# If I use a small n (les than ~500) the code never hangs 

class TestInfiniteLoop(unittest.TestCase):
    def test_infinite_loop(self):
        cu = CU('gpu') # or 'gpu' if you have CUDA
        with closing(cu):
            n = np.int32(1000)          # Number of particles 

            chunks = np.int32(5)        # Number of chunks
            chunk_size = n/chunks       # Chunk size
            eps = np.float64(1e-10)     # Epsilon (avoids r=0 if sources=targets)

            # Pointer to first value of each chunk
            offset_particles = np.arange(0, n+n/chunks, n/chunks, dtype=np.int32)
            # Hand coded interaction list: 0->0,1; 1->0,1,2; 2->1,2,3; 3->2,3,4; 4->3,4
            inter_list = np.array([0,1, 0,1,2, 1,2,3, 2,3,4, 3,4], dtype=np.int32)
            # Pointer to first value of each chunk in inter_list array
            offset_chunks = np.array([0,2,5,8,11,13], dtype=np.int32)
            # Pointers to particles (only a range for this problem)
            sort_particles = np.arange(n, dtype=np.int32)

            # input arrays
            # Sources
            xj = np.random.random_sample(n);  yj = np.random.random_sample(n); zj = np.random.random_sample(n)
            m = np.random.random_sample(n)
            # Targets
            xi = np.random.random_sample(n); yi = np.random.random_sample(n); zi = np.random.random_sample(n)

            # output arrays
            phi = np.zeros(n, dtype=np.float64)

            tic = time.time()
            # tag the arrays
            xiDev = cu.input(xi); yiDev = cu.input(yi); ziDev = cu.input(zi)
            xjDev = cu.input(xj); yjDev = cu.input(yj); zjDev = cu.input(zj); mDev = cu.input(m)
            offParDev = cu.input(offset_particles)
            intListDev = cu.input(inter_list)
            offChunDev = cu.input(offset_chunks)
            sizeDev = cu.input(np.array([chunk_size],dtype=np.int32))
            sortDev = cu.input(sort_particles)
            epsDev = cu.input(np.array([eps], dtype=np.float64))
            dRes  = cu.output(phi)

            cu.enqueue(nbody, ntid=dRes.size, args=(xiDev, yiDev, ziDev, xjDev, yjDev, zjDev, mDev, dRes,
                                                    sizeDev, offParDev, intListDev, offChunDev, sortDev, epsDev)) 

            cu.wait() # synchronize

            toc = time.time()

            # Python code for verification
            test = np.zeros(len(xi)) 
            for i in range(chunks):
                CI_start = offset_particles[i]
                CI_end   = offset_particles[i+1]
                chunkId  = CI_start/chunk_size
                CJ_start = offset_chunks[chunkId]
                CJ_end   = offset_chunks[chunkId+1]
                for CJ in inter_list[CJ_start:CJ_end]:
                    for j in sort_particles[offset_particles[CJ]:offset_particles[CJ+1]]:
                        dx = xi[CI_start:CI_end]-xj[j]
                        dy = yi[CI_start:CI_end]-yj[j]
                        dz = zi[CI_start:CI_end]-zj[j]
                        test[CI_start:CI_end] += m[j]/np.sqrt(dx*dx+dy*dy+dz*dz+eps)
            
            print(np.allclose(test, phi)) # verify

        total_time = toc - tic
        print 'Time: %f'%total_time

if __name__ == '__main__':
    unittest.main()
