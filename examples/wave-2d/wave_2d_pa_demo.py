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
import numpy as np
from numba import njit, stencil
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    raise RuntimeError("matplotlib is needed to run this example. Try 'conda install matplotlib'")

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

class State(object):
    def __init__(self, speed, s):
        self.speed = speed
        self.s = s

        self.t = 0.0

        self.s = s
        self.s2  = self.s // 2
        self.s4  = self.s // 4
        self.s8  = self.s // 8
        self.s16 = self.s // 16

        self.p = np.zeros((self.s, self.s)) # past
        self.c = np.zeros((self.s, self.s)) # current
        self.f = np.zeros((self.s, self.s)) # future

        self.dt = 0.0001     # Time resolution of the simulation
        self.dx = 0.01       # Distance between elements
        self.r = self.speed * self.dt / self.dx

        self.n = 300
        self.last = -1

        for i in range(self.s2 - self.s16 - 1, self.s2 + self.s16):
            # Initial conditions
            self.c[i, self.s2 - self.s16 - 1 : self.s2 + self.s16 ] = - 2 * np.cos(0.5 * 2 * np.pi / (self.s8) * np.arange(self.s2 - self.s16 , self.s2 + self.s16 + 1)) * np.cos(0.5 * 2 * np.pi / (self.s8) * i)
            self.p[i, 0:self.s] = self.c[i, 0:self.s]

        self.im = plt.imshow(self.c, animated=True)

    def step(self):
        # Wave equation
        runWaveStencil(self.p, self.c, self.f, self.r, self.t, self.n, self.s2, self.s4, self.s)

        # Transparent corner handling
        self.f[0, 0] =     (2 * self.c[0, 0]     + (self.r-1) * self.p[0, 0]     + 2*self.r*self.r* (self.c[1, 0]       + self.c[0, 1]       - 2*self.c[0, 0]))     / (1+self.r) # X:1; Y:1
        self.f[self.s-1, self.s-1] = (2 * self.c[self.s-1, self.s-1] + (self.r-1) * self.p[self.s-1, self.s-1] + 2*self.r*self.r* (self.c[self.s-2, self.s-1] + self.c[self.s-1, self.s-2] - 2*self.c[self.s-1, self.s-1])) / (1+self.r) # X:s; Y:s
        self.f[0, self.s-1] =   (2 * self.c[0, self.s-1]   + (self.r-1) * self.p[0, self.s-1]   + 2*self.r*self.r* (self.c[1, self.s-1]     + self.c[0, self.s-2]   - 2*self.c[0, self.s-1]))   / (1+self.r) # X:1; Y:s
        self.f[self.s-1, 0] =   (2 * self.c[self.s-1, 0]   + (self.r-1) * self.p[self.s-1, 0]   + 2*self.r*self.r* (self.c[self.s-2, 0]   + self.c[self.s-1, 1]     - 2*self.c[self.s-1, 0]))   / (1+self.r) # X:s; Y:1

        # Rotate buffers for next iteration
        tmp = self.p
        self.p = self.c
        self.c = self.f
        self.f = tmp

        self.t += self.dt
        self.im.set_array(self.c)

    def plot_step(self, x):
        if x > self.last:
            print("plot_step", x)
            self.step()
            self.last = x

def wave2d():
    speed = 10      # Propagation speed
    s = 512         # Array size (spatial resolution of the simulation)

    fig = plt.figure()
    state = State(speed, s)
    ani = FuncAnimation(fig, state.plot_step)
    plt.show()

    return True

def main(*args):
    wave2d()
    
if __name__ == "__main__":
    main(*sys.argv[1:])

