from numba import jit
import numpy as np
import math
import time

@jit('f4[:,:](i2,f4[:,:])')
def ra_numba(doy, lat):
    
    M, N = lat.shape
    
    ra = np.zeros_like(lat)
    Gsc = 0.0820
    
    # math.pi doesnt work?
    # NumbaError: 11:31: Binary operations mul on values typed object_ and object_ not (yet) supported)
    pi = math.pi
    #pi = 3.1415926535897932384626433832795

    dr = 1 + 0.033 * math.cos( 2 * pi / 365 * doy)
    decl = 0.409 * math.sin( 2 * pi / 365 * doy - 1.39 )
    
    for i in range(M):
        for j in range(N):
            
            # it crashes without the float() wrapped around the array slicing?!
            ws = math.acos(-1 * math.tan(float(lat[i,j])) * math.tan(decl))
            ra[i,j] = 24 * 60 / pi * Gsc * dr * ( ws * math.sin(float(lat[i,j])) * math.sin(decl) + math.cos(float(lat[i,j])) * math.cos(decl) * math.sin(ws)) * 11.6

    
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
dims = []

for dim in [25,50,100,200,400,800,1600]:
    
    dims.append(dim)
    
    lat = np.deg2rad(np.ones((dim,dim), dtype=np.float32) * 45.) # array of 45 degrees latitude converted to rad
    
    tic = time.clock()
    ra_nb = ra_numba(doy, lat)
    numb.append(time.clock() - tic)

    tic = time.clock()
    ra_np = ra_numpy(doy, lat)
    nump.append(time.clock() - tic)
    
    tic = time.clock()
    ra_py = ra_python(doy, lat)
    py.append(time.clock() - tic)
    
dims = np.array(dims)**2
py = np.array(py)
numb = np.array(numb)
nump = np.array(nump)
