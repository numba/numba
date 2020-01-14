from numba import njit, gdb
import numpy as np

@njit(parallel={'spirv':True})
def f1(a, b):
    c = a + b
    return c

global_size = 50, 1
local_size = 32, 1, 1
N = global_size[0] * local_size[0]
print("N", N)

a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
#a = np.array(np.random.random(N), dtype=np.float32)
#b = np.array(np.random.random(N), dtype=np.float32)
#c = np.ones_like(a)

print("a:", a, hex(a.ctypes.data))
print("b:", b, hex(b.ctypes.data))
#print("c:", c, hex(c.ctypes.data))
c = f1(a,b)
print("BIG RESULT c:", c, hex(c.ctypes.data))
for i in range(N):
    if c[i] != 2.0:
        print("First index not equal to 2.0 was", i)
        break
