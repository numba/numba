import numpy as np
import numba

print('Numpy:', np.__version__)
print('Numba:', numba.__version__)

vector = np.zeros(3)

def f(s, vector):
    vector[0] = s
    return vector

print('\ns as scalar')
s = 3.14
print(f(s, vector)) # works
try:
    numba.njit(f)(s, vector) # works
    print('Numba works')
except:
    print('numba.njit(f)(s, vector) - Does not work')

print('\ns as 0d-np.array')
s = np.array(3.14)
print(f(s, vector)) # works
try:
    numba.njit(f)(s, vector) # does not work
    print('Numba works')
except:
    print('numba.njit(f)(s, vector) - Does not work')

print('\ns as 1d-np.array of length 1')
s = np.array([3.14])
print(f(s, vector)) # works
try:
    numba.njit(f)(s, vector) # does not work
    print('Numba works')
except:
    print('numba.njit(f)(s, vector) - Does not work')

print('\ns as 1d-np.array of length 1, indexed at 0')
s = np.array([3.14])
print(f(s[0], vector)) # works
try:
    numba.njit(f)(s[0], vector) # works
    print('Numba works')
except:
    print('numba.njit(f)(s[0], vector) - Does not work')