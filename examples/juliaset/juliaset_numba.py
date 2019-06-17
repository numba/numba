#! /usr/bin/env python

#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from __future__ import print_function

import sys
import time
import os

import numpy as np
import numpy.matlib
import numba

# This code was ported from a MATLAB implementation available at
# http://www.albertostrumia.it/Fractals/FractalMatlab/Julia/juliaSH.m.
# Original MATLAB implementation is copyright (c) the author.

@numba.njit()
def iterate(col, Z, c):
    for k in range(col):
        Z = Z*Z + c
    return np.exp(-np.abs(Z))

def juliaset(iters):
    col = 128             # color depth
    m = 5000              # image size
    cx = 0                # center X
    cy = 0                # center Y
    l = 1.5               # span
    zoomAmount = 0.6

    # The c constant.
    c = -.745429 + .11308j

    for zoom in range(iters):
        # `x` and `y` are two 1000-element arrays representing the x
        # and y axes: [-1.5, -1.497, ..., 0, ..., 1.497, 1.5] on the
        # first iteration of this loop.
        x = np.linspace(cx-l, cx+l, m)
        y = np.linspace(cy-l, cy+l, m)

        # `X` and `Y` are two arrays containing, respectively, the x-
        # and y-coordinates of each point on a 1000x1000 grid.
        (X, Y) = np.meshgrid(x, y)

        # Let `Z` represent the complex plane: a 1000x1000 array of
        # numbers each with a real and a complex part.
        Z = X + Y*1j

        # Iterate the Julia set computation (squaring each element of
        # Z and adding c) for `col` steps.
        W = iterate(col, Z, c)

        # Mask out the NaN values (overflow).
        minval = np.nanmin(W)
        W[np.isnan(W)] = minval - minval/10
        print("checksum W = ", W.sum())

        # Zoom into the next frame, shrinking the distance that `x`
        # and `y` will cover.
        l = l * zoomAmount

def main (*args):
    tstart = time.time()
    iterate(1, np.empty((1,2), dtype=complex), complex(0.0))
    htime = time.time() - tstart
    print("SELFPRIMED ", htime)

    tstart = time.time()
    juliaset(10)
    htime = time.time() - tstart
    print("SELFTIMED ", htime)

if __name__ == "__main__":
    main(*sys.argv[1:])
