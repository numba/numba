from numba import *
from numba.tests import test_support
import numpy as np
import math
import unittest
import time
import logging


def ra_numba(doy, lat):
    '''Modified from http://nbviewer.ipython.org/4117896/'''
    M, N = lat.shape

    ra = np.zeros_like(lat)   
    Gsc = 0.0820

    pi = math.pi

    dr = 1 + 0.033 * math.cos( 2 * pi / 365 * doy)
    decl = 0.409 * math.sin( 2 * pi / 365 * doy - 1.39 )

    for i in range(M):
        for j in range(N):
            ws = math.acos(-1 * math.tan(lat[i,j]) * math.tan(decl))
            ra[i,j] = 24 * 60 / pi * Gsc * dr * (
                ws * math.sin(lat[i,j]) * math.sin(decl) +
                math.cos(lat[i,j]) * math.cos(decl) *
                math.sin(ws)) * 11.6

    return ra


def ra_numpy(doy, lat):
    Gsc = 0.0820
    pi = math.pi

    dr = 1 + 0.033 * np.cos( 2 * pi / 365 * doy)
    decl = 0.409 * np.sin( 2 * pi / 365 * doy - 1.39 )
    ws = np.arccos(-np.tan(lat) * np.tan(decl))

    ra = 24 * 60 / pi * Gsc * dr * (
        ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) *
        np.sin(ws)) * 11.6

    return ra


class TestIssue57(unittest.TestCase):
    def test_ra_numba(self):
        test_fn = jit('f4[:,:](i2,f4[:,:])')(ra_numba)
        lat = np.deg2rad(np.ones((1000, 1000), dtype=np.float32) * 45.)
        t0 = time.time()
        control_arr = ra_numpy(120, lat)
        t1 = time.time()
        test_arr = test_fn(120, lat)
        t2 = time.time()
        logging.info('Numpy time %0.6fs, Numba time %0.6fs' % (
                t1 - t0, t2 - t1))
        self.assertTrue(np.allclose(test_arr, control_arr))


if __name__ == "__main__":

    test_support.main()
