
import time
import numpy as np
import simdtest

def print_array_info(name, array):
    print "%s stride: %d data: %s" % (repr(name), array.strides[0], repr(array.data))


def test_adds(element_count, repeat = 100):
    arg0 = np.linspace(0.0, 1000.0, element_count).astype(np.float32)
    arg1 = np.linspace(1000.0, 0.0, element_count).astype(np.float32)
    result = np.ndarray(element_count, dtype=np.float32)

    def do_test(func):
        start = time.time()
        func(arg0, arg1, result)
        end = time.time()
        return end - start

    tests = [ 'scalar_add', 'vvm_add', 'simd_add', 'rsimd_add', 'faith_add' ]

    print '\n====== test adds for %d elements (%d times) ======' % (element_count, repeat)
    for i in tests:
        f = simdtest.__dict__[i]
        results = [ do_test(f) for x in range(0, repeat) ]
        total = sum(results)
        print '%s elapsed (%d runs) total: %f s avg: %f ms max: %f ms min: %f ms' % (i, repeat, total, total/repeat * 1000.0, max(results) * 1000.0, min(results) * 1000.0) 

def test_sins(element_count, repeat = 100):
    arg0 = np.linspace(0.0, np.pi, element_count).astype(np.float32)
    result = np.ndarray(element_count, dtype=np.float32)

    def do_test(func):
        start = time.time()
        func(arg0, result)
        end = time.time()
        return end - start

    tests = [ 'scalar_sin', 'vvm_sin', 'rvvm_sin' ]
    print '\n====== test sins for %d elements (%d times) ======' % (element_count, repeat)
    for i in tests:
        f = simdtest.__dict__[i]
        results = [ do_test(f) for x in range(0,repeat) ]
        total = sum(results)
        print '%s elapsed (%d runs) total: %f s avg: %f ms max: %f ms min: %f ms' % (i, repeat, total, total/repeat * 1000.0, max(results) * 1000.0, min(results) * 1000.0) 

test_adds(2*1024)
test_adds(256*1024)
test_adds(64*1024*1024, 20)

test_sins(2*1024)
test_sins(256*1025)
test_sins(64*1024*1024, 20)
