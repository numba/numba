import psutil
from numba import (njit, jitclass, uint32)
import numpy as np
from numba.runtime.nrt import rtsys
from numba.unsafe.refcount import dump_refcount

@jitclass([
    ('attr', uint32),
])
class JitClass:
    def __init__(self):
        pass


# @njit  # same with njit
# def f():
#     cs = [JitClass() for i in range(1000)]

#     # leak
#     for i, c in enumerate(cs):
#         # dump_refcount(c)
#         c.attr = 1234

#     # No enumerate, no leak
#     # for c in cs:
#     #     c.attr = 1234
# p = psutil.Process()
# for _ in range(100000):
#     f()
#     # leak proportional to the size of cs
#     print("memory used by program:", {p.memory_info().rss / 1e6}," MB")


@njit
def inner(cs):
    total = 0
    dump_refcount(cs)
    it = enumerate(cs)
    dump_refcount(cs)
    for i, v in it:
        dump_refcount(cs)
        dump_refcount(v)
    dump_refcount(cs)
    return total

@njit  # same with njit
def f():
    cs = (np.ones(10),) # np.ones(11), np.ones(11))

    dump_refcount(cs)
    print("HERE")
    inner(cs)
    print("END")
    dump_refcount(cs)
    # No enumerate, no leak
    # for c in cs:
    #     c.attr = 1234
print(rtsys.get_allocation_stats())

f()


print(rtsys.get_allocation_stats())


# print(inner.inspect_types())