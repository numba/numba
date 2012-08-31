#


from profutils import *
import numpy as np
import numba as nb
import numexpr as ne

from numbapro.vectorize.basic import BasicVectorize
from numbapro.vectorize.cuda import CudaVectorize
from numbapro.vectorize.parallel import ParallelVectorize
from numbapro.vectorize.stream import StreamVectorize

def compute_Y(r, g, b):
    return 0.299*r + 0.114*b + 0.587*g

def numpy_code(r, g, b, result):
    result = 0.299*r + 0.114*b + 0.587*g

def numexpr_code(r, g, b, result):
    ne.evaluate('0.299*r + 0.114*b + 0.587*g', out = result)


# code based on formulas found in http://en.wikipedia.org/wiki/Error_function
# max error = 1.5e-7
def aprox_erf(x):
    if (x < 0):
        sign = 0.0 - 1.0
        x = 0.0 - x
    else:
        sign = 1.0

    p = 0.3275911
    a1 = 0.254829592
    a2 = 0.284496736
    a3 = 1.421413741
    a4 = 1.453152027
    a5 = 1.061405429
    t = 1.0/(1.0 + p * x)
    res = 1.0 - 1.0/(1.0 + a1*t - a2*(t*t) + a3*(t*t*t) - a4*(t*t*t*t) + a5*(t*t*t*t*t))
    return sign * res

def numexpr_aprox_erf(x, res):
    p = 0.3275911
    a1 = 0.254829592
    a2 = 0.284496736
    a3 = 1.421413741
    a4 = 1.453152027
    a5 = 1.061405429
    sign = np.sign(x)
    t = ne.evaluate('1.0/(1.0 + p*abs(x))')
    ne.evaluate('(1.0 - 1.0/(1.0 + a1*t - a2*(t**2) + a3*(t**3) - a4*(t**4) + a5*(t**5)))', out = res)

def numexpr_mad_aprox_erf(x, res):
    p = 0.3275911
    a1 = 0.254829592
    a2 = 0.284496736
    a3 = 1.421413741
    a4 = 1.453152027
    a5 = 1.061405429
    ne.evaluate('where(x >=0 , 1.0 - 1.0/(1.0 + a1*(1.0/(1.0 + p*abs(x))) - a2*((1.0/(1.0 + p*abs(x)))**2) + a3*((1.0/(1.0 + p*abs(x)))**3) - a4*((1.0/(1.0 + p*abs(x)))**4) + a5*((1.0/(1.0 + p*abs(x)))**5)), -(1.0 - 1.0/(1.0 + a1*(1.0/(1.0 + p*abs(x))) - a2*((1.0/(1.0 + p*abs(x)))**2) + a3*((1.0/(1.0 + p*abs(x)))**3) - a4*((1.0/(1.0 + p*abs(x)))**4) + a5*((1.0/(1.0 + p*abs(x)))**5))))', out = res)

def numexpr_mad_pos_aprox_erf(x, res):
    p = 0.3275911
    a1 = 0.254829592
    a2 = 0.284496736
    a3 = 1.421413741
    a4 = 1.453152027
    a5 = 1.061405429
    ne.evaluate('where(x>=0, 1.0, -1.0) - where(x>=0, 1.0, -1.0)/(1.0 + a1*(1.0/(1.0 + p*abs(x))) - a2*((1.0/(1.0 + p*abs(x)))**2) + a3*((1.0/(1.0 + p*abs(x)))**3) - a4*((1.0/(1.0 + p*abs(x)))**4) + a5*((1.0/(1.0 + p*abs(x)))**5))', out = res)

def numpy_aprox_erf(x, res):
    sign = np.sign(x)
    x = np.abs(x)

    p = 0.3275911
    a1 = 0.254829592
    a2 = 0.284496736
    a3 = 1.421413741
    a4 = 1.453152027
    a5 = 1.061405429
    t = 1.0/(1.0 + p * x)
    res = sign * (1.0 - 1.0/(1.0 + a1*t - a2*(t**2) + a3*(t**3) - a4*(t**4) + a5*(t**5)))


def run_computeY_AoS(elements):
    def build_ufunc(kind, func, type):
        builder = kind(func)
        builder.add(ret_type = type, arg_types = [type] * 3)
        return builder.build_ufunc()

    dataset = np.random.random((elements, 3))
    result  = np.zeros(elements)
    test_args = [dataset[:,0], dataset[:,1], dataset[:,2], result]
    bv_ufunc = build_ufunc(BasicVectorize, compute_Y, nb.d)
#   cv_ufunc = build_ufunc(CudaVectorize, compute_Y, nb.d)
    pv_ufunc = build_ufunc(ParallelVectorize, compute_Y, nb.d)
    sv_ufunc = build_ufunc(StreamVectorize, compute_Y, nb.d)
    return profile_functions([('nbp-basic', bv_ufunc, test_args),
#                             ('nbp-cuda',  cv_ufunc, test_args),
                              ('nbp-stream', pv_ufunc, test_args),
                              ('nbp-parallel', sv_ufunc, test_args),
                              ('numpy', numpy_code, test_args),
                              ('numexpr', numexpr_code, test_args)])

def run_computeY_SoA(elements):
    def build_ufunc(kind, func, type):
        builder = kind(func)
        builder.add(ret_type = type, arg_types = [type] * 3)
        return builder.build_ufunc()

    dataset = np.random.random((3, elements))
    result  = np.zeros(elements)
    test_args = [dataset[0,:], dataset[1,:], dataset[2,:], result]
    bv_ufunc = build_ufunc(BasicVectorize, compute_Y, nb.d)
#   cv_ufunc = build_ufunc(CudaVectorize, compute_Y, nb.d)
    pv_ufunc = build_ufunc(ParallelVectorize, compute_Y, nb.d)
    sv_ufunc = build_ufunc(StreamVectorize, compute_Y, nb.d)
    return profile_functions([('nbp-basic', bv_ufunc, test_args),
#                             ('nbp-cuda',  cv_ufunc, test_args),
                              ('nbp-stream', pv_ufunc, test_args),
                              ('nbp-parallel', sv_ufunc, test_args),
                              ('numpy', numpy_code, test_args),
                              ('numexpr', numexpr_code, test_args)])

def run_aprox_erf(elements):
    def build_ufunc(kind, func, type):
        builder = kind(func)
        builder.add(ret_type = type, arg_types = [ type ])
        return builder.build_ufunc()

    dataset = np.random.random(elements)
    result = np.zeros(elements)
    test_args = [ dataset, result ]
    bv_ufunc = build_ufunc(BasicVectorize, aprox_erf, nb.d)
#   cv_ufunc = build_ufunc(CudaVectorize, aprox_erf, nb.d)
    pv_ufunc = build_ufunc(ParallelVectorize, aprox_erf, nb.d)
    sv_ufunc = build_ufunc(StreamVectorize, aprox_erf, nb.d)
    return profile_functions([
            ('numpy', numpy_aprox_erf, test_args),
            ('numexpr', numexpr_aprox_erf, test_args),
            ('numexpr-crazy', numexpr_mad_aprox_erf, test_args),
            ('numexpr-crazy-pos', numexpr_mad_pos_aprox_erf, test_args),
            ('nbp-basic', bv_ufunc, test_args),
#            ('nbp-cuda',  cv_ufunc, test_args),
            ('nbp-stream', pv_ufunc, test_args),
            ('nbp-parallel', sv_ufunc, test_args)
            ])

    

if __name__ == '__main__':
    print '==== compute Y AoS'
    print_profile_results(run_computeY_AoS(1920 * 1080 * 10))
    print '==== compute Y SoA'
    print_profile_results(run_computeY_SoA(1920 * 1080 * 10))
    print '==== erf aproximation'
    print_profile_results(run_aprox_erf(1024 * 1024 * 32))
