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

from numba import njit, stencil
try:
    from PIL import Image
except ImportError:
    raise RuntimeError("Pillow is needed to run this example. Try 'conda install pillow'")

@stencil()
def xsten(a):
    return ((a[-1,-1] * -1.0) + (a[-1,0] * -2.0) + (a[-1,1] * -1.0) + a[1,-1] + (a[1,0] * 2.0) + a[1,1]) / 12.0

@stencil()
def ysten(a):
    return ((a[-1,-1] * -1.0) + (a[0,-1] * -2.0) + (a[1,-1] * -1.0) + a[-1,1] + (a[0,1] * 2.0) + a[1,1]) / 12.0

@stencil()
def harris_common(a):
    return (a[-1,-1] + a[-1,0] + a[-1,1] + a[0,-1] + a[0,0] + a[0,1] + a[1,-1] + a[1,0] + a[1,1])

@njit()
def harris(Iin):
    Ix = xsten(Iin)
    Iy = ysten(Iin)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    Sxx = harris_common(Ixx)
    Syy = harris_common(Iyy)
    Sxy = harris_common(Ixy)
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy
    return det - (0.04 * trace * trace)

def main (*args):
    iterations = 10
    
    if len(args) > 0:
        input_file = args[0]
    else:
        raise ValueError("A jpeg file must be provided as the first command line parameter.")

    parts = os.path.splitext(input_file)
    new_file_name = parts[0] + "-corners" + parts[1]

    input_img = Image.open(input_file).convert('L')
    input_arr = np.array(input_img)
    
    tstart = time.time()
    for i in range(iterations):
        output_arr = harris(input_arr)
    htime = time.time() - tstart
    print("SELFTIMED ", htime)

    new_img = Image.fromarray(output_arr.astype(np.uint8), mode=input_img.mode)
    new_img.format = input_img.format
    new_img.save(new_file_name)

if __name__ == "__main__":
    main(*sys.argv[1:])
