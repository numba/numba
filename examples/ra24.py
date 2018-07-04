#!/usr/bin/env python
from numba import jit
import numpy as np
import math
import time


@jit
def ra_numba(doy, lat):
    ra = np.zeros_like(lat)
    Gsc = 0.0820
    
    pi = math.pi

    dr = 1 + 0.033 * math.cos( 2 * pi / 365 * doy)
    decl = 0.409 * math.sin( 2 * pi / 365 * doy - 1.39 )
    tan_decl = math.tan(decl)
    cos_decl = math.cos(decl)
    sin_decl = math.sin(decl)
    
    for idx, latval in np.ndenumerate(lat):
        ws = math.acos(-math.tan(latval) * tan_decl)
        ra[idx] = 24 * 60 / pi * Gsc * dr * ( ws * math.sin(latval) * sin_decl + math.cos(latval) * cos_decl * math.sin(ws)) * 11.6

    return ra


def ra_numpy(doy, lat):
    Gsc = 0.0820

    pi = math.pi
    
    dr = 1 + 0.033 * np.cos( 2 * pi / 365 * doy)
    decl = 0.409 * np.sin( 2 * pi / 365 * doy - 1.39 )
    ws = np.arccos(-np.tan(lat) * np.tan(decl))
    
    ra = 24 * 60 / pi * Gsc * dr * ( ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws)) * 11.6
    
    return ra


ra_python = ra_numba.py_func

doy = 120 # day of year

py = []
nump = []
numb = []

for dim in [50, 100, 400, 1600]:
    lat = np.deg2rad(np.ones((dim,dim), dtype=np.float32) * 45.) # array of 45 degrees latitude converted to rad

    # JIT warmup
    ra_numba(doy, lat)

    tic = time.clock()
    ra_nb = ra_numba(doy, lat)
    numb.append(time.clock() - tic)

    tic = time.clock()
    ra_np = ra_numpy(doy, lat)
    nump.append(time.clock() - tic)
    
    tic = time.clock()
    ra_py = ra_python(doy, lat)
    py.append(time.clock() - tic)
    

print("pure Python times:", py)
print("Numpy times:", nump)
print("Numba times:", numb)
