#! /usr/bin/env python
from __future__ import print_function

import sys
import time
import os

import numpy as np

from numba import njit, stencil
from PIL import Image


@stencil()
def gaussian_blur(a):
    return (a[-2,-2] * 0.003  + a[-1,-2] * 0.0133 + a[0,-2] * 0.0219 + a[1,-2] * 0.0133 + a[2,-2] * 0.0030 +
            a[-2,-1] * 0.0133 + a[-1,-1] * 0.0596 + a[0,-1] * 0.0983 + a[1,-1] * 0.0596 + a[2,-1] * 0.0133 +
            a[-2, 0] * 0.0219 + a[-1, 0] * 0.0983 + a[0, 0] * 0.1621 + a[1, 0] * 0.0983 + a[2, 0] * 0.0219 +
            a[-2, 1] * 0.0133 + a[-1, 1] * 0.0596 + a[0, 1] * 0.0983 + a[1, 1] * 0.0596 + a[2, 1] * 0.0133 +
            a[-2, 2] * 0.003  + a[-1, 2] * 0.0133 + a[0, 2] * 0.0219 + a[1, 2] * 0.0133 + a[2, 2] * 0.0030)

#@njit()
@njit(parallel=True)
def run_gaussian_blur(input_arr, iterations):
    sinput_arr = gaussian_blur(input_arr)
    for i in range(iterations-1):
        output_arr = gaussian_blur(sinput_arr)
        sinput_arr, output_arr = output_arr, sinput_arr

    return sinput_arr

def main (*args):
    iterations = 1
    input_file = "sample.jpg" 
    
    if len(args) > 0:
        input_file = args[0]

    if len(args) > 1:
        iterations = int(args[1])

    parts = os.path.splitext(input_file)
    new_file_name = parts[0] + "-blur" + parts[1]

    input_img = Image.open(input_file)
    input_arr = np.array(input_img)
    output_arr = run_gaussian_blur(input_arr, iterations).astype(input_arr.dtype)

    new_img = Image.fromarray(output_arr, mode=input_img.mode)
    new_img.format = input_img.format
    new_img.save(new_file_name)

if __name__ == "__main__":
    main(*sys.argv[1:])
