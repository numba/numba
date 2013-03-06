import numpy as np

from numbapro import cuda
import api

curand = api.libcurand()
print curand, curand.dll

N = 10
ary32 = np.zeros(N, dtype=np.float32)
ary64 = np.zeros(N, dtype=np.float64)
devary32 = cuda.to_device(ary32)
devary64 = cuda.to_device(ary64)

print 'curand version', api.libcurand().version
rndgen = api.Generator(api.CURAND_RNG_PSEUDO_DEFAULT)
rndgen.set_pseudo_random_generator_seed(1234)

rndgen.generate_uniform(devary32, N)
rndgen.generate_uniform(devary64, N)

devary32.to_host()
devary64.to_host()

print ary32
print ary64


rndgen.generate_normal(devary32, N, 0, 1)
rndgen.generate_normal(devary64, N, 0, 1)

devary32.to_host()
devary64.to_host()

print ary32
print ary64



rndgen.generate_log_normal(devary32, N, 0, 1)
rndgen.generate_log_normal(devary64, N, 0, 1)

devary32.to_host()
devary64.to_host()

print ary32
print ary64

