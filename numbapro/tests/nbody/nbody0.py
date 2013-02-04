# Based upon the version modified by Mark Harris

import numpy as np
import time
import numbapro
from numba import *
import unittest

flopsPerInteraction = 30
SOFTENING_SQUARED = 0.01

def normalize (vector):
    dist = np.sqrt((vector * vector).sum())
    if dist > 1e-6:
        vector /= dist
    return dist


def randomize_bodies(pos, vel, cluster_scale, velocity_scale, n):
    np.random.seed(42)
    scale = cluster_scale
    vscale = scale * velocity_scale
    inner = 2.5 * scale
    outer = 4.0 * scale

    i = 0
    while i < n:
        point = np.random.random(3) / 2.
        length = normalize(point)
        if length > 1.:
            continue
        pos[i,:3] = point * ((inner + (outer - inner)) * np.random.random(3))
        pos[i,3] = 1.0
        axis = np.array((0., 0., 1.))
        normalize(axis)
        if (1 - ((point * axis).sum())) < 1e-6:
            axis[0] = point[1]
            axis[1] = point[0]
            normalize(axis)
        vv = np.cross(pos[i,:3], axis)
        vel[i,:3] = vscale * vv
        vel[i,3]  = 1.0
        #print("%d: %s, %s" % (i, pos[i], vel[i]))
        i += 1


def check_correctness(pin, pout, v, dt, n, integrate_0, integrate_1):
    pin_ref = np.zeros_like(pin)
    pout_ref = np.zeros_like(pout)
    v_ref = np.zeros_like(v)
    randomize_bodies(pin_ref, v_ref, 1.54, 8.0, n)
    integrate_0(pout_ref, pin, np.copy(v), dt, n)
    integrate_1(pout,     pin, np.copy(v), dt, n)
 
    errt = 0
    errmax = 0

    errs = np.fabs(pout_ref - pout).reshape(4 * n)
    errt = errs.sum()
    errmax = errs.max()

    print("Maximum error: %0.4f -- Total error: %0.4f" % (errmax, errt))
    return errt

def check_overflow(x):
    return np.isnan(np.sum(x))        

def body_body_interaction(force, pos_mass0, pos_mass1):
    r = pos_mass1[:3] - pos_mass0[:3]
    dist_sqr = (r * r).sum()
    dist_sqr += SOFTENING_SQUARED
    inv_dist = 1.0 / np.sqrt(dist_sqr)
    inv_dist_cube = inv_dist * inv_dist * inv_dist
    s = pos_mass1[3] * inv_dist_cube
    force += r * s

def integrate(position_out, position_in, velocity, delta_time, n):
    for i in range(n):
        p = position_in[i]
        f = np.zeros(3)
        for j in range(n):
            body_body_interaction(f, p, position_in[j])
        inv_mass = velocity[i,3]
        v = velocity[i,:3] + f * inv_mass * delta_time
        position_out[i,:3] = p[:3] + v * delta_time
        position_out[i,3] = position_in[i,3]
        velocity[i,:3] = v

@autojit
def fast_body_body_interaction(force, pos_mass0, pos_mass1):
    r = pos_mass1[:3] - pos_mass0[:3]
    dist_sqr = (r * r).sum()
    dist_sqr += SOFTENING_SQUARED
    inv_dist = 1.0 / np.sqrt(dist_sqr)
    inv_dist_cube = inv_dist * inv_dist * inv_dist
    s = pos_mass1[3] * inv_dist_cube
    force += r * s



@autojit
def fast_integrate(position_out, position_in, velocity, delta_time, n):
    for i in range(n):
        # ------------
        # original does not work
        #   p = position_in[i]
        p = position_in[i, :] + 0 # don't like array assignment?
        # ------------
        f = np.zeros(3, dtype=np.double)
        for j in range(n):
            # ------------
            # original does not work
            #   pj = position_in[j]
            pj = position_in[j, :] + 0 # don't like array assignment?
            # ------------
            fast_body_body_interaction(f, p, pj)
        inv_mass = velocity[i,3]
        v = velocity[i,:3] + f * inv_mass * delta_time
        position_out[i,:3] = p[:3] + v * delta_time
        position_out[i,3] = position_in[i,3]
        velocity[i,:3] = v

def compute_perf_stats(milliseconds, iterations, n):
    interactionsPerSecond = float(n * n)
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds
    return interactionsPerSecond * flopsPerInteraction;


class TestNBody0(unittest.TestCase):
    def test(self):
        n = 128
        dt = 0.01667

        pin = np.zeros((n, 4))
        pout = np.zeros((n, 4))
        v = np.zeros((n, 4))

        randomize_bodies(pin, v, 1.54, 8.0, n)

        errt = check_correctness(pin, pout, v, dt, n, integrate, fast_integrate)
        self.assertEqual(errt, 0, "Array expression is causing errors.")

        #    time0 = time.time()
        #    for i in range(iterations):
        #        fast_integrate(pout, pin, v, dt, n)
        #        pin, pout = pout, pin
        #    time1 = time.time()
        #    ms = (time1 - time0)*1000
        #    gf = compute_perf_stats(ms, iterations, n)
        #
        #    print("%d n-body iterations" % iterations)
        #    print("%f ms: %f GFLOP/s" % (ms, gf))


if __name__ == "__main__":
#    TestNBody0('test').debug()
    unittest.main()
