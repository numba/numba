from math import log, sin, cos
import time
import numpy as np
import npufunc

x = np.linspace(0.5, 0.9, 1<<16)
print x

S = time.time()
ans = npufunc.logit(x)
E = time.time()
print E - S


gold = np.vectorize(lambda x: log(x/(1-x))+sin(x)+2*cos(x))(x)

print ans
print gold

for i, (x, y) in enumerate(zip(ans, gold)):
    err = abs(x-y)/y
    assert err < 1e-8, "%d | x = %f | y = %f | err = %f" % (i, x, y, err)
