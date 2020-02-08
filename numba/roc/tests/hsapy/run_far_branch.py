import math

import numpy as np

import numba


def run_far_jump():

    gt_as_str = 'float32'
    R_EARTH = 6371.0  # km

    @numba.roc.jit(device=True)
    def deg2rad(deg):
        return math.pi * deg / 180.0

    sig = '%s(%s, %s, %s, %s)' % ((gt_as_str,) * 5)

    @numba.vectorize(sig, target='roc')
    def gpu_great_circle_distance(lat1, lng1, lat2, lng2):
        '''Return the great-circle distance in km between (lat1, lng1) and (lat2, lng2)
        on the surface of the Earth.'''
        lat1, lng1 = deg2rad(lat1), deg2rad(lng1)
        lat2, lng2 = deg2rad(lat2), deg2rad(lng2)

        sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
        sin_lat2, cos_lat2 = math.sin(lat2), math.cos(lat2)

        delta = lng1 - lng2
        sin_delta, cos_delta = math.sin(delta), math.cos(delta)

        numerator = math.sqrt((cos_lat1 * sin_delta)**2 +
                                (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta)**2)
        denominator = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta
        return R_EARTH * math.atan2(numerator, denominator)

    arr = np.random.random(10).astype(np.float32)

    gpu_great_circle_distance(arr, arr, arr, arr)



if __name__ == '__main__':
    run_far_jump()
