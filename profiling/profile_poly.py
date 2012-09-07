'''
Test overhead of vectorize
'''

import numpy as np
import time

import utils

import pyximport; pyximport.install()
import cypoly, nbpoly, nbpropoly

UDT = [('cython', cypoly.poly_d),
       ('numba',  nbpoly.poly_d),
       ('basic',  nbpropoly.b_poly_d),
       ('stream',  nbpropoly.s_poly_d),
       ('parallel',  nbpropoly.p_poly_d),]

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
            C = A.copy()
            D = np.zeros(N, dtype=np.double)

            timings = utils.run_timing(function, A, B, C, D)

            assert (np.sqrt(B**2 + 4 * A * C) == D).all()

            fastest = np.min(timings)
            wordpersecond = N/fastest
            record[name].append(wordpersecond)

except MemoryError:
    pass
finally:
    import pickle
    with open('poly_d_bandwidth.dat', 'wb') as fout:
        pickle.dump(record, fout)

