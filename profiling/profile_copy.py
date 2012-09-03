'''
Test overhead of vectorize
'''

import numpy as np
import time

import utils

import pyximport; pyximport.install()
import cycopy, nbcopy, nbprocopy

UDT = [('cython', cycopy.copy_d),
       ('numba',  nbcopy.copy_d),
       ('basic',  nbprocopy.b_copy_d),
       ('stream',  nbprocopy.s_copy_d),
       ('parallel',  nbprocopy.p_copy_d),]

record = {}
for name, _ in UDT:
    record[name] = []
try:
    for power in range(28):
        N = 2 ** power
        print 'N\t%s' % N
        for name, function in UDT:
            print name
            A = np.arange(N, dtype=np.double)
            B = np.zeros(N)

            timings = utils.run_timing(function, A, B)

            assert (A == B).all()

            fastest = np.min(timings)
            wordpersecond = N/fastest
            record[name].append(wordpersecond)

except MemoryError:
    pass
finally:
    import pickle
    with open('copy_d_bandwidth.dat', 'wb') as fout:
        pickle.dump(record, fout)

