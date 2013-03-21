# Example from Travis Oliphant

import ctypes as ct
import numpy.random as nr
import os.path
from numpy.distutils.misc_util import get_shared_lib_extension

mtrand = ct.CDLL(nr.mtrand.__file__)

# Should we parse this from randomkit.h in the numpy directory? 
RK_STATE_LEN = len(nr.get_state()[1])

class rk_state(ct.Structure):
    _fields_ = [("key", ct.c_ulong * RK_STATE_LEN),
               ("pos", ct.c_int),
               ("has_gauss", ct.c_int),
               ("gauss", ct.c_double),
               ("has_binomial", ct.c_int),
               ("psave", ct.c_double),
               ("nsave", ct.c_long),
               ("r", ct.c_double),
               ("q", ct.c_double),
               ("fm", ct.c_double),
               ("m", ct.c_long),
               ("p1", ct.c_double),
               ("xm", ct.c_double),
               ("xl", ct.c_double),
               ("xr", ct.c_double),               
               ("c", ct.c_double),
               ("laml", ct.c_double),
               ("lamr", ct.c_double),
               ("p2", ct.c_double),
               ("p3", ct.c_double),
               ("p4", ct.c_double)]

try:
    rk_randomseed = mtrand.rk_randomseed
    rk_seed = mtrand.rk_seed
    rk_gamma = mtrand.rk_gamma
    rk_normal = mtrand.rk_normal
except AttributeError as e:
    raise ImportError(str(e))

rk_randomseed.argtypes = [ct.POINTER(rk_state)]

rk_seed.restype = None
rk_seed.argtypes = [ct.c_long, ct.POINTER(rk_state)]

state = rk_state()
state_p = ct.pointer(state)
state_vp = ct.cast(state_p, ct.c_void_p)


rk_gamma.restype = ct.c_double
rk_gamma.argtypes = [ct.POINTER(rk_state), ct.c_double, ct.c_double]

rk_normal.restype = ct.c_double
rk_normal.argtypes = [ct.POINTER(rk_state), ct.c_double, ct.c_double]

def init():
    if rk_randomseed(state_p) != 0:
        raise ValueError("Cannot initialize the random number generator.")

init()
