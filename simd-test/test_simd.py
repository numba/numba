
import time
import numpy as np
import simdtest

def print_array_info(name, array):
    print "%s stride: %d data: %s" % (repr(name), array.strides[0], repr(array.data))


arg0 = np.linspace(0.0, 1000.0, 1024*1024*64).astype(np.float32)
arg1 = np.linspace(1000.0, 0.0, 1024*1024*64).astype(np.float32)
result = np.ndarray(1024*1024*64, dtype=np.float32)
print_array_info('arg0', arg0)
print_array_info('arg1', arg1)
print_array_info('result', result)


def do_test(func):
    start = time.time()
    simdtest.__dict__[func](arg0, arg1, result)
    end = time.time()
    return end - start

tests = [ 'scalar_add', 'vvm_add', 'simd_add', 'rsimd_add', 'faith_add' ]

for i in tests:
    results = [ do_test(i) for x in range(0, 100) ]
    print '%s elapsed avg: %f max: %f min: %f' % (i, sum(results)/len(results), max(results), min(results)) 


