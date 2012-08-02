import time
import numpy as np
import npufunc

def pyprimetest(x):
    from math import sqrt, ceil
    if x == 2:
        return True
    if x % 2 == 0:
        return False
    for i in range(3, x, 2): #int(ceil(sqrt(x)))):
        if x % i == 0:
            return False
    return True

x = np.arange(2, 2**16, dtype=np.int32)

S = time.time()
ans = npufunc.primetest(x)
E = time.time()
print(E - S)

if False:
    gold = np.vectorize(pyprimetest)(x)
    for i, (x, y) in enumerate(zip(ans, gold)):
        assert x==y, i
