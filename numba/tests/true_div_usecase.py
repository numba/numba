from __future__ import division


# This function has its own module in order to be compiled with the right
# __future__ flag (and be tested alongside the 2.x legacy division operator).

def truediv_usecase(x, y):
    return x / y

