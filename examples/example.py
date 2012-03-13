from math import sin, pi
from numba.decorators import vectorize

#@vectorize
def sinc(x):
    if x==0.0:
        return 1.0
    else:
        return sin(x*pi)/(pi*x)

from numba.translate import Translate
t = Translate(sinc)
t.translate()
print t.mod
res = t.make_ufunc()

#sinc = vectorize(sinc)

from numpy import linspace
print res(linspace(-5,5,50))
