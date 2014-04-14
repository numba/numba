# -*- coding: utf-8 -*-
"""
Example for closures. Closures may be of arbitrary dept, and they keep
the scope alive as long as the closure is alive. Only variables that are
closed over (cell variables in the defining function, free variables in the
closure), are kept alive. See also numba/tests/closures/test_closure.py
"""
from __future__ import print_function, division, absolute_import

from numba import autojit, jit, float_
from numpy import linspace

@jit
def generate_power_func(n):
    
    @jit(float_(float_))
    def nth_power(x):
        return x ** n

    # This is a native call
    print(nth_power(10))

    # Return closure and keep all cell variables alive
    return nth_power

for n in range(2, 5):
    func = generate_power_func(n)
    print([func(x) for x in linspace(1.,2.,10.)])
