
def exercise1(x):
	return 0.25*x*x*x + 0.75*x*x - 1.5*x - 2

def exercise1a(x):
	return ((0.25*x+0.75)*x-1.5)*x - 2

from translate import Translate
t0 = Translate(exercise1)
t0.translate()
print t0.mod
exercise1 = t0.make_ufunc()

t1 = Translate(exercise1a)
t1.translate()
print t1.mod
exercise1a = t1.make_ufunc()


"""
# make entry block

# make true block
# make false block
# make join block

"""
    
