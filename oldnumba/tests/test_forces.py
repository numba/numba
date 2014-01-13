# issue: #68
# Thanks to tpievila

from numba import *

import numpy as np
from numpy.random import randn
from numpy import zeros, double

nple = 64
k_fene = 15.0
R0 = 2.0
R0_2 = R0*R0

sigma = 1.0
sigma2 = sigma*sigma
sigma6 = sigma2*sigma2*sigma2
sigma12 = sigma6*sigma6

dpart = 1.12246*sigma
rcut = dpart
r2cut = dpart*dpart
k_fene = 15.0
force = 5.0 # Constant bias force in ext_force
width = 0.8775  # for ext force

@autojit()
def ext_force1(x, y, extx, exty):
    for i in xrange(nple):
        if abs(x[i]) > rcut:
            extx[i] = 0.0
        elif abs(y[i]) > rcut + width:
            r2dist = 1.0 / (x[i] * x[i])
            r6dist = r2dist * r2dist * r2dist
            extx[i] = (x[i] * 48.0 * r6dist * r2dist *
                       (sigma12 * r6dist - 0.5 * sigma6))
        else:
            ydista = y[i] - width - rcut
            ydistb = y[i] + width + rcut
            dist2a = x[i] * x[i] + ydista * ydista
            dist2b = x[i] * x[i] + ydistb * ydistb
            if dist2b < r2cut or dist2a < r2cut:
                if dist2b < r2cut:
                    ydist = ydistb
                    r2dist = 1.0 / dist2b
                else:
                    ydist = ydista
                    r2dist = 1.0 / dist2a
                r6dist = r2dist * r2dist * r2dist
                lj_factor = 48.0 * r6dist * r2dist * (sigma12 * r6dist - 0.5 * sigma6)
                extx[i] = x[i] * lj_factor
                exty[i] = ydist * lj_factor
        if abs(x[i]) < 0.5 and abs(y[i]) < width + rcut:
            extx[i] += force  # constant bias force in the x-direction
    return extx, exty


@autojit()
def ext_force2(x, y):
    extx = object_(zeros(nple, double))
    exty = object_(zeros(nple, double))
    ext_force1(x, y, extx, exty)
    return extx, exty


def test_forces():
    extx = zeros(nple, double)
    exty = zeros(nple, double)

    x, y = randn(nple), randn(nple)
    x1, y1 = ext_force1(x, y, extx, exty)
    x2, y2 = ext_force2(x, y)

    assert np.allclose(x1, x2), x2 - x1
    assert np.allclose(y1, y2), y2 - y1

if __name__ == '__main__':
    test_forces()
