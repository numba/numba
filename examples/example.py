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
sinc = t.make_ufunc()

#sinc = vectorize(sinc)

from numpy import linspace
x = linspace(-5,5,1001)
y = sinc(x)
from pylab import plot, show
plot(x,y)
show()
