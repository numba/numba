#    Maximum error: 0.0000 -- Total error: 0.0000
#    10 n-body iterations
#    588.119984 ms: 0.008357 GFLOP/s

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
def numbapro_body_body_interaction_aryexpr(force, pos_mass0, pos_mass1):
    r = pos_mass1[:3] - pos_mass0[:3]
    dist_sqr = (r * r).sum()
    for i in range(3):
        r[i] = pos_mass1[i] - pos_mass0[i]
        dist_sqr += r[i] ** 2
    # -----------
    dist_sqr += SOFTENING_SQUARED
    inv_dist = 1.0 / np.sqrt(dist_sqr)
    inv_dist_cube = inv_dist * inv_dist * inv_dist
    s = pos_mass1[3] * inv_dist_cube
    force += r * s

@autojit
def numbapro_integrate_aryexpr(position_out, position_in, velocity, delta_time, n):
    for i in range(n):
        # ------------
        # original does not work
        #   p = position_in[i]
        p = position_in[i, :] + 0 # don't like array assignment?
        # ------------
        f = np.zeros(3, dtype=np.double) # should warn user about missing dtype
        for j in range(n):
            # ------------
            # original does not work
            #   pj = position_in[j]
            pj = position_in[j, :] + 0 # don't like array assignment?
            # ------------
            numbapro_body_body_interaction_aryexpr(f, p, pj)
        inv_mass = velocity[i,3]

        v = velocity[i,:3] + f * inv_mass * delta_time
        position_out[i,:3] = p[:3] + v * delta_time
        position_out[i,3] = position_in[i,3]
        velocity[i,:3] = v


@autojit
def numbapro_body_body_interaction(force, pos_mass0, pos_mass1):
    # -----------
    # original is slower
    #   r = pos_mass1[:3] - pos_mass0[:3]
    #   dist_sqr = (r * r).sum()
    dist_sqr = 0.0
    r = np.zeros(3, dtype=np.double)
    for i in range(3):
        r[i] = pos_mass1[i] - pos_mass0[i]
        dist_sqr += r[i] ** 2
    # -----------
    dist_sqr += SOFTENING_SQUARED
    inv_dist = 1.0 / np.sqrt(dist_sqr)
    inv_dist_cube = inv_dist * inv_dist * inv_dist
    s = pos_mass1[3] * inv_dist_cube

    # -----------
    # original is slower and generate more errors
    #   force += r * s
    for i in range(3):
        force[i] += r[i] * s
    # -----------

@autojit
def numbapro_integrate(position_out, position_in, velocity, delta_time, n):
    for i in range(n):
        # ------------
        # original does not work
        #   p = position_in[i]
        p = position_in[i, :] + 0 # don't like array assignment?
        # ------------
        f = np.zeros(3, dtype=np.double) # should warn user about missing dtype
        for j in range(n):
            # ------------
            # original does not work
            #   pj = position_in[j]
            pj = position_in[j, :] + 0 # don't like array assignment?
            # ------------
            numbapro_body_body_interaction(f, p, pj)
        inv_mass = velocity[i,3]

        # --------------
        # original is slower
        #        v = velocity[i,:3] + f * inv_mass * delta_time
        #        position_out[i,:3] = p[:3] + v * delta_time
        #        position_out[i,3] = position_in[i,3]
        #        velocity[i,:3] = v
        for k in range(3):
            v = velocity[i, k] + f[k] * inv_mass * delta_time
            position_out[i, k] = p[k] + v * delta_time
            velocity[i, k] = v
        position_out[i, 3] = position_in[i, 3]
        # --------------



@autojit
def numba_body_body_interaction(force, pos_mass, p1, p2):
    dist_sqr = 0.0
    r = np.zeros(3, dtype=np.double)  # numba requires type info
    for i in range(3):
        r[i] = pos_mass[p2, i] - pos_mass[p1, i]
        dist_sqr += r[i] ** 2
    dist_sqr += SOFTENING_SQUARED
    inv_dist = 1.0 / np.sqrt(dist_sqr)
    inv_dist_cube = inv_dist * inv_dist * inv_dist
    s = pos_mass[p1, 3] * inv_dist_cube
    for i in range(3):
        force[i] += r[i] * s

@autojit
def numba_integrate(position_out, position_in, velocity, delta_time, n):
    for i in range(n):
        f = np.zeros(3, dtype=np.double) # numba requires type info
        for j in range(n):
            numba_body_body_interaction(f, position_in, i, j)
        inv_mass = velocity[i,3]
        for k in range(3):
            v = velocity[i, k] + f[k] * inv_mass * delta_time
            position_out[i, k] = position_in[i, k] + v * delta_time
            velocity[i, k] = v
        position_out[i, 3] = position_in[i, 3]

def compute_perf_stats(milliseconds, iterations, n):
    interactionsPerSecond = float(n * n)
    interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds
    return interactionsPerSecond * flopsPerInteraction;


class TestNbody1(unittest.TestCase):
    def test(self):
        n = 128
        iterations = 10
        dt = 0.01667

        pin = np.zeros((n, 4))
        pout = np.zeros((n, 4))
        v = np.zeros((n, 4))

        randomize_bodies(pin, v, 1.54, 8.0, n)

        errt_numba = check_correctness(pin, pout, v, dt, n, integrate,
                                       numba_integrate)
        errt_numbapro = check_correctness(pin, pout, v, dt, n, integrate,
                                          numbapro_integrate)
        errt_numbapro_aryexpr = check_correctness(pin, pout, v, dt, n,
                                                  integrate,
                                                  numbapro_integrate_aryexpr)

        print 'Numba'.center(40, '=')
        time0 = time.time()
        for i in range(iterations):
            numba_integrate(pout, pin, v, dt, n)
            pin, pout = pout, pin
        time1 = time.time()
        ms = (time1 - time0)*1000
        gf_numba = compute_perf_stats(ms, iterations, n)

        print("%d n-body iterations" % iterations)
        print("%f ms: %f GFLOP/s" % (ms, gf_numba))



        print 'NumbaPro'.center(40, '=')
        time0 = time.time()
        for i in range(iterations):
            numbapro_integrate(pout, pin, v, dt, n)
            pin, pout = pout, pin
        time1 = time.time()
        ms = (time1 - time0)*1000
        gf_numbapro = compute_perf_stats(ms, iterations, n)

        print("%d n-body iterations" % iterations)
        print("%f ms: %f GFLOP/s" % (ms, gf_numbapro))



        print 'NumbaPro+AryExpr'.center(40, '=')
        time0 = time.time()
        for i in range(iterations):
            numbapro_integrate_aryexpr(pout, pin, v, dt, n)
            pin, pout = pout, pin
        time1 = time.time()
        ms = (time1 - time0)*1000
        gf_numbapro_aryexpr = compute_perf_stats(ms, iterations, n)

        print("%d n-body iterations" % iterations)
        print("%f ms: %f GFLOP/s" % (ms, gf_numbapro_aryexpr))

        # There should not be any error
        self.assertEqual(errt_numba, 0)
        self.assertEqual(errt_numbapro, 0)
        self.assertEqual(errt_numbapro_aryexpr, 0)
        # NumbaPro cannot be slower.
        self.assertTrue(gf_numbapro_aryexpr >= gf_numbapro,
                               "Array-expression is slower than simple loops")
        self.assertTrue(gf_numbapro >= gf_numba,
                               "NumbaPro is slower than Numba")


if __name__ == "__main__":
    unittest.main()
