from math import sin, pi

def sinc(x):
    if x==0:
        return 1.0
    else:
        return sin(x*pi)/(pi*x)

from translate import Translate
t = Translate(sinc)
t.translate()
print t.mod
res = t.make_ufunc()

def myfunc(a):
	return a+a+a+a

t0 = Translate(myfunc)
t0.translate()
newfunc = t0.make_ufunc()

"""
# make entry block

# make true block
# make false block
# make join block

"""
    
