from numba import autojit, jit, float_
from numpy import linspace

@autojit
def generate_power_func(n):
    @jit(float_(float_))
    def nth_power(x):
        return x ** n

    return nth_power

for n in range(2, 5):
    func = generate_power_func(n)
    print [func(x) for x in linspace(1.,2.,10.)]
