#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

# This code was ported from a Julia implementation as part of the 
# ParallelAccelerator Julia package which in turn was ported from a
# MATLAB implementation available at
# https://www.piso.at/julius/programming/octave-2d-wave-equation
# with permission of the author.
# Original MATLAB implementation is copyright (c) 2014 Julius Piso.

from __future__ import print_function

import sys
import time
import numpy as np
from numba import njit, stencil

@stencil()
def rws_kernel(p, c, r):
    return 2 * c[0, 0] - p[0, 0] + r * r * (c[-1, 0] + c[1, 0] + c[0, -1] + c[0, 1] - 4*c[0, 0])

@njit(parallel=True)
def runWaveStencil(p, c, f, r, t, n, s2, s4, s):
    rws_kernel(p, c, r, out=f)

    # Dynamic source
    f[s2+s4-2:s2+s4+1, 0:2] = 1.5 * np.sin(t*n)
    f[s2-s4-2:s2-s4+1, 0:2] = 1.5 * np.sin(t*n)
    f[1:s-1, 0:2] = 1.0

    # Transparent boundary handling
    f[1:s-1, 0]   = (2.0 * c[1:s-1, 0]   + (r-1.0) * p[1:s-1, 0]   + 2.0*r*r*(c[1:s-1, 1]   + c[2:s, 0]   + c[0:s-2, 0]   - 3.0 * c[1:s-1, 0]))   / (1.0+r) # Y:1
    f[1:s-1, s-1] = (2.0 * c[1:s-1, s-1] + (r-1.0) * p[1:s-1, s-1] + 2.0*r*r*(c[1:s-1, s-2] + c[2:s, s-1] + c[0:s-2, s-1] - 3.0 * c[1:s-1, s-1])) / (1.0+r) # Y:s
    f[0, 1:s-1]   = (2.0 * c[0, 1:s-1]   + (r-1.0) * p[0, 1:s-1]   + 2.0*r*r*(c[1, 1:s-1]   + c[0, 2:s]   + c[0, 0:s-2]   - 3.0 * c[0, 1:s-1]))   / (1.0+r) # X:1
    f[s-1, 1:s-1] = (2.0 * c[s-1, 1:s-1] + (r-1.0) * p[s-1, 1:s-1] + 2.0*r*r*(c[s-2, 1:s-1] + c[s-1, 2:s] + c[s-1, 0:s-2] - 3.0 * c[s-1, 1:s-1])) / (1.0+r) # Y:s

def prime_wave2d():
    speed = 10         # Propagation speed
    s = 16             # Array size (spatial resolution of the simulation)

    s2  = s // 2
    s4  = s // 4
    s8  = s // 8
    s16 = s // 16

    p = np.zeros((s, s)) # past
    c = np.zeros((s, s)) # current
    f = np.zeros((s, s)) # future

    dt = 0.0001     # Time resolution of the simulation
    dx = 0.01       # Distance between elements
    r = speed * dt / dx

    n = 300

    for i in range(s2 - s16 - 1, s2 + s16):
        # Initial conditions
        c[i, s2 - s16 - 1 : s2 + s16 ] = - 2 * np.cos(0.5 * 2 * np.pi / (s8) * np.arange(s2 - s16 , s2 + s16 + 1)) * np.cos(0.5 * 2 * np.pi / (s8) * i)
        p[i, 0:s] = c[i, 0:s]

    runWaveStencil(p, c, f, r, 0, n, s2, s4, s)

def wave2d():
    speed = 10      # Propagation speed
    s = 6000        # Array size (spatial resolution of the simulation)

    s2  = s // 2
    s4  = s // 4
    s8  = s // 8
    s16 = s // 16

    p = np.zeros((s, s)) # past
    c = np.zeros((s, s)) # current
    f = np.zeros((s, s)) # future

    dt = 0.0001     # Time resolution of the simulation
    dx = 0.01       # Distance between elements
    r = speed * dt / dx

    n = 300

    for i in range(s2 - s16 - 1, s2 + s16):
        # Initial conditions
        c[i, s2 - s16 - 1 : s2 + s16 ] = - 2 * np.cos(0.5 * 2 * np.pi / (s8) * np.arange(s2 - s16 , s2 + s16 + 1)) * np.cos(0.5 * 2 * np.pi / (s8) * i)
        p[i, 0:s] = c[i, 0:s]

    stopTime = 0.05

    # Main loop
    t = 0.0
    while t <= stopTime:
        # Wave equation
        runWaveStencil(p, c, f, r, t, n, s2, s4, s)

        # Transparent corner handling
        f[0:1, 0:1] =     (2 * c[0:1, 0:1]     + (r-1) * p[0:1, 0:1]     + 2*r*r* (c[1:2, 0:1]       + c[0:1, 1:2]       - 2*c[0:1, 0:1]))     / (1+r) # X:1; Y:1
        f[s-1:s, s-1:s] = (2 * c[s-1:s, s-1:s] + (r-1) * p[s-1:s, s-1:s] + 2*r*r* (c[s-2:s-1, s-1:s] + c[s-1:s, s-2:s-1] - 2*c[s-1:s, s-1:s])) / (1+r) # X:s; Y:s
        f[0:1, s-1:s] =   (2 * c[0:1, s-1:s]   + (r-1) * p[0:1, s-1:s]   + 2*r*r* (c[1:2, s-1:s]     + c[0:1, s-2:s-1]   - 2*c[0:1, s-1:s]))   / (1+r) # X:1; Y:s
        f[s-1:s, 0:1] =   (2 * c[s-1:s, 0:1]   + (r-1) * p[s-1:s, 0:1]   + 2*r*r* (c[s-2:s-1, 0:1]   + c[s-1:s, 1:2]     - 2*c[s-1:s, 0:1]))   / (1+r) # X:s; Y:1

        # Rotate buffers for next iteration
        tmp = p
        p = c
        c = f
        f = tmp

        t += dt

    return f

def main(*args):
    prime_wave2d()

    tstart = time.time()
    wave2d()
    htime = time.time() - tstart
    print("SELFTIMED ", htime)
    
if __name__ == "__main__":
    main(*sys.argv[1:])

