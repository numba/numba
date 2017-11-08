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

from numba import jit, stencil
try:
    from PIL import Image
except ImportError:
    raise RuntimeError("Pillow is needed to run this example. Try 'conda install pillow'")

@stencil()
def gaussian_blur(a):
    return (a[-2,-2] * 0.003  + a[-1,-2] * 0.0133 + a[0,-2] * 0.0219 + a[1,-2] * 0.0133 + a[2,-2] * 0.0030 +
            a[-2,-1] * 0.0133 + a[-1,-1] * 0.0596 + a[0,-1] * 0.0983 + a[1,-1] * 0.0596 + a[2,-1] * 0.0133 +
            a[-2, 0] * 0.0219 + a[-1, 0] * 0.0983 + a[0, 0] * 0.1621 + a[1, 0] * 0.0983 + a[2, 0] * 0.0219 +
            a[-2, 1] * 0.0133 + a[-1, 1] * 0.0596 + a[0, 1] * 0.0983 + a[1, 1] * 0.0596 + a[2, 1] * 0.0133 +
            a[-2, 2] * 0.003  + a[-1, 2] * 0.0133 + a[0, 2] * 0.0219 + a[1, 2] * 0.0133 + a[2, 2] * 0.0030)

def gaussian_blur_std(a, res):
    ashape = a.shape
    for i in range(2,ashape[0]-2):
        for j in range(2,ashape[1]-2):
            res[i,j] = (a[i-2,j-2] * 0.003  + a[i-1,j-2] * 0.0133 + a[i,j-2] * 0.0219 + a[i+1,j-2] * 0.0133 + a[i+2,j-2] * 0.0030 +
                        a[i-2,j-1] * 0.0133 + a[i-1,j-1] * 0.0596 + a[i,j-1] * 0.0983 + a[i+1,j-1] * 0.0596 + a[i+2,j-1] * 0.0133 +
                        a[i-2,j+0] * 0.0219 + a[i-1,j+0] * 0.0983 + a[i,j+0] * 0.1621 + a[i+1,j+0] * 0.0983 + a[i+2,j+0] * 0.0219 +
                        a[i-2,j+1] * 0.0133 + a[i-1,j+1] * 0.0596 + a[i,j+1] * 0.0983 + a[i+1,j+1] * 0.0596 + a[i+2,j+1] * 0.0133 +
                        a[i-2,j+2] * 0.003  + a[i-1,j+2] * 0.0133 + a[i,j+2] * 0.0219 + a[i+1,j+2] * 0.0133 + a[i+2,j+2] * 0.0030)
    return res

@stencil()
def gaussian_blur_3d(a):
    return (a[-2,-2,0] * 0.003  + a[-1,-2,0] * 0.0133 + a[0,-2,0] * 0.0219 + a[1,-2,0] * 0.0133 + a[2,-2,0] * 0.0030 +
            a[-2,-1,0] * 0.0133 + a[-1,-1,0] * 0.0596 + a[0,-1,0] * 0.0983 + a[1,-1,0] * 0.0596 + a[2,-1,0] * 0.0133 +
            a[-2, 0,0] * 0.0219 + a[-1, 0,0] * 0.0983 + a[0, 0,0] * 0.1621 + a[1, 0,0] * 0.0983 + a[2, 0,0] * 0.0219 +
            a[-2, 1,0] * 0.0133 + a[-1, 1,0] * 0.0596 + a[0, 1,0] * 0.0983 + a[1, 1,0] * 0.0596 + a[2, 1,0] * 0.0133 +
            a[-2, 2,0] * 0.003  + a[-1, 2,0] * 0.0133 + a[0, 2,0] * 0.0219 + a[1, 2,0] * 0.0133 + a[2, 2,0] * 0.0030)

def gaussian_blur_std_3d(a, res):
    ashape = a.shape
    for i in range(2,ashape[0]-2):
        for j in range(2,ashape[1]-2):
            for k in range(ashape[2]):
                res[i,j,k] = (a[i-2,j-2,k] * 0.003  + a[i-1,j-2,k] * 0.0133 + a[i,j-2,k] * 0.0219 + a[i+1,j-2,k] * 0.0133 + a[i+2,j-2,k] * 0.0030 +
                              a[i-2,j-1,k] * 0.0133 + a[i-1,j-1,k] * 0.0596 + a[i,j-1,k] * 0.0983 + a[i+1,j-1,k] * 0.0596 + a[i+2,j-1,k] * 0.0133 +
                              a[i-2,j+0,k] * 0.0219 + a[i-1,j+0,k] * 0.0983 + a[i,j+0,k] * 0.1621 + a[i+1,j+0,k] * 0.0983 + a[i+2,j+0,k] * 0.0219 +
                              a[i-2,j+1,k] * 0.0133 + a[i-1,j+1,k] * 0.0596 + a[i,j+1,k] * 0.0983 + a[i+1,j+1,k] * 0.0596 + a[i+2,j+1,k] * 0.0133 +
                              a[i-2,j+2,k] * 0.003  + a[i-1,j+2,k] * 0.0133 + a[i,j+2,k] * 0.0219 + a[i+1,j+2,k] * 0.0133 + a[i+2,j+2,k] * 0.0030)
    return res
def main (*args):
    iterations = 10
    
    if len(args) > 0:
        input_file = args[0]
    else:
        raise ValueError("A jpeg file must be provided as the first command line parameter.")

    if len(args) > 1:
        iterations = int(args[1])

    parts = os.path.splitext(input_file)
    new_file_name = parts[0] + "-blur" + parts[1]

    input_img = Image.open(input_file)
    input_arr = np.array(input_img)

    assert(input_arr.ndim == 2 or input_arr.ndim == 3)

    tstart = time.time()
    if input_arr.ndim == 2:
        output_arr = gaussian_blur(input_arr)
    else:
        output_arr = gaussian_blur_3d(input_arr)

    htime = time.time() - tstart
    print("SELFPRIMED ", htime)

    tstart = time.time()
    for i in range(iterations):
        if input_arr.ndim == 2:
            output_arr = gaussian_blur(input_arr)
        else:
            output_arr = gaussian_blur_3d(input_arr)
        input_arr, output_arr = output_arr, input_arr
    htime = time.time() - tstart
    print("SELFTIMED ", htime)

    output_arr = input_arr.astype(np.uint8)

    new_img = Image.fromarray(output_arr, mode=input_img.mode)
    new_img.format = input_img.format
    new_img.save(new_file_name)
    input_img.close()

    input_img = Image.open(input_file)
    input_arr = np.array(input_img)

    tstart = time.time()
    output_arr = input_arr.copy()
    for i in range(iterations):
        if input_arr.ndim == 2:
            gaussian_blur_std(input_arr, output_arr)
        else:
            gaussian_blur_std_3d(input_arr, output_arr)
        input_arr, output_arr = output_arr, input_arr
    htime = time.time() - tstart
    print("Standard Python time", htime)
    input_img.close()

if __name__ == "__main__":
    main(*sys.argv[1:])
