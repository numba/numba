'''
Test overhead of vectorize
'''

import numpy as np
import time

import utils

import pyximport; pyximport.install()
import cyadd, nbadd, nbproadd

UDT = [('cython', cyadd.add_d),
       ('numba',  nbadd.add_d),
       ('basic',  nbproadd.b_add_d),
       ('stream',  nbproadd.s_add_d),
       ('parallel',  nbproadd.p_add_d),]

record = {}
for name, _ in UDT:
    record[name] = []
try:
    for power in range(26):
        N = 2 ** power
        print 'N\t%s' % N
        for name, function in UDT:
            print name
            A = np.arange(N, dtype=np.double)
            B = A.copy()
            C = np.zeros(N, dtype=np.double)

            timings = utils.run_timing(function, A, B, C)

            assert (A + B == C).all()

            fastest = np.min(timings)
            wordpersecond = N/fastest
            record[name].append(wordpersecond)

except MemoryError:
    pass
finally:
    import pickle
    with open('add_d_bandwidth.dat', 'wb') as fout:
        pickle.dump(record, fout)

