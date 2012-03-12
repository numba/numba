
def exercise1(x):
	return 0.25*x**3 + 0.75*x**2 - 1.5*x - 2

def exercise1a(x):
	return ((0.25*x+0.75)*x-1.5)*x - 2

from numba.translate import Translate
t = Translate(exercise1)
t.translate()
print t.mod
res = t.make_ufunc()
   
