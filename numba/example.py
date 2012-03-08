from math import sin, pi

def sinc(x):
    if x==0.0:
        return 1.0
    else:
        return sin(x*pi)/(pi*x)

from translate import Translate
t = Translate(sinc)
t.translate()
print t.mod
res = t.make_ufunc()

from numpy import linspace
print res(linspace(-5,5,50))

"""
# make entry block

# make true block
# make false block
# make join block

"""
    
