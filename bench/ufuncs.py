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


def build_ufunc(kind, func, type):
    builder = kind(func)
    builder.add(ret_type = type, arg_types = [type] * 3)
    return builder.build_ufunc()

def numpy_code(r, g, b, result):
    result = 0.299*r + 0.114*b + 0.587*g

def numexpr_code(r, g, b, result):
    ne.evaluate('0.299*r + 0.114*b + 0.587*g', out = result)

def run_bench(elements):
    dataset = np.random.random((elements, 3))
    result  = np.zeros(elements)
    test_args = [dataset[:,0], dataset[:,1], dataset[:,2], result]
    bv_ufunc = build_ufunc(BasicVectorize, compute_Y, nb.d)
#    cv_ufunc = build_ufunc(CudaVectorize, compute_Y, nb.d)
    pv_ufunc = build_ufunc(ParallelVectorize, compute_Y, nb.d)
    sv_ufunc = build_ufunc(StreamVectorize, compute_Y, nb.d)
    return profile_functions([('nbp-basic', bv_ufunc, test_args),
 #                             ('nbp-cuda',  cv_ufunc, test_args),
                              ('nbp-stream', pv_ufunc, test_args),
                              ('nbp-parallel', sv_ufunc, test_args),
                              ('numpy', numpy_code, test_args),
                              ('numexpr', numexpr_code, test_args)])

def run_bench2(elements):
    dataset = np.random.random((3, elements))
    result  = np.zeros(elements)
    test_args = [dataset[0,:], dataset[1,:], dataset[2,:], result]
    bv_ufunc = build_ufunc(BasicVectorize, compute_Y, nb.d)
#    cv_ufunc = build_ufunc(CudaVectorize, compute_Y, nb.d)
    pv_ufunc = build_ufunc(ParallelVectorize, compute_Y, nb.d)
    sv_ufunc = build_ufunc(StreamVectorize, compute_Y, nb.d)
    return profile_functions([('nbp-basic', bv_ufunc, test_args),
 #                             ('nbp-cuda',  cv_ufunc, test_args),
                              ('nbp-stream', pv_ufunc, test_args),
                              ('nbp-parallel', sv_ufunc, test_args),
                              ('numpy', numpy_code, test_args),
                              ('numexpr', numexpr_code, test_args)])
if __name__ == '__main__':
#    ne.set_num_threads(4)
    print "==== AoS"
    print_profile_results(run_bench(1920 * 1080 * 10))
    print "==== SoA"
    print_profile_results(run_bench2(1920 * 1080 * 10))
