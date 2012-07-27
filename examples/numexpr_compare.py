from numba.decorators import vectorize

@vectorize
def exercise1(x):
	return 0.25*x**3 + 0.75*x**2 - 1.5*x - 2

@vectorize
def exercise1a(x):
	return ((0.25*x+0.75)*x-1.5)*x - 2

  
