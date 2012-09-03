#
# Test of kernel performing transform of vectors by quaternions in a vectored way
#


import numpy as np
from profutils import *
import numba as nb
from numba.decorators import jit
from numbapro.vectorize.gufunc import GUFuncVectorize


def hl_quat_normalize(q0):
    return q0/np.sqrt(np.sum(q0.q0))

def hl_quat_product(q0, q1):
    result = np.empty_like(q0)
    quat_product(q0,q1, result)
    return result

def hl_quat_conj(q0):
    result = np.empty_like(q0)
    quat_conj(q0, result)
    return result

def hl_quat_from_vect(v):
    result = np.empty((4,), dtype = v.dtype)
    for i in range(3):
        result[i] = v[i]
    result[3] = 0
    return result

def hl_vect_from_quat(q):
    return np.copy(q[0:3])
    

def hl_vect_by_quat(v, q0):
    # sandwich product
    return hl_vect_from_quat(hl_quat_product(hl_quat_product(q0, hl_quat_from_vect(v)), hl_quat_conj(q0)))

def quat_normalize(q0, result):
    norm = np.sqrt(np.sum(q0*q0))
    result = q0 / norm

# quats assumed to be in ijkw form
def quat_product(q0, q1, result):
    result[0] = q0[3] * q1[0] + q0[0] * q1[3] + q0[1] * q1[2] - q0[2] * q1[1] # i
    result[1] = q0[3] * q1[1] - q0[0] * q1[2] + q0[1] * q1[3] + q0[2] * q1[0] # j
    result[2] = q0[3] * q1[2] + q0[0] * q1[1] - q0[1] * q1[0] + q0[2] * q1[3] # k
    result[3] = q0[3] * q1[3] - q0[0] * q1[0] - q0[1] * q1[1] - q0[2] * q1[2] # re

def quat_conj(q0, result):
    result[0] = q0[0]
    result[1] = q0[1]
    result[2] = q0[2]
    result[3] = 0.0 - result[3]

def quat_from_vect(v, result):
    for i in range(3):
        result[i] = v[i]
    result[3] = 0.0

def vect_from_quat(q, result):
    result=q[0:3]

def vect_by_quat(v, q0, result):
    vq = np.empty((4,), dtype = v.dtype)
    r0 = np.empty((4,), dtype = v.dtype)
    r1 = np.empty((4,), dtype = v.dtype)
    quat_from_vect(v, vq)
    quat_product(q0, vq, r0)
    quat_conj(q0, r1)
    quat_product(r0, r1, vq)
    vect_from_quat(vq, v)
    
def vect_array_by_quats_inlined(vs, q0, result):
    N = vs.shape[0]
    for i in range(N):
        tmp0 = q0[3] * vs[i,0]                   + q0[1] * vs[i,2] - q0[2] * vs[i,1] # i
        tmp1 = q0[3] * vs[i,1] - q0[0] * vs[i,2]                   + q0[2] * vs[i,0] # j
        tmp2 = q0[3] * vs[i,2] + q0[0] * vs[i,1] - q0[1] * vs[i,0]                   # k
        tmp3 =             0.0 - q0[0] * vs[i,0] - q0[1] * vs[i,1] - q0[2] * vs[i,2] # re

        result[i,0] = tmp0 * q0[3] - tmp3 * q0[0] + tmp1 * q0[2] - tmp2 * q0[1] # i
        result[i,1] = tmp3 * q0[1] - tmp0 * q0[2] + tmp1 * q0[3] - tmp2 * q0[0] # j
        result[i,2] = tmp3 * q0[2] + tmp0 * q0[1] + tmp1 * q0[0] + tmp2 * q0[3] # k

def vect_array_by_quats(array, q0, result):
    N = array.shape[0]
    for i in range(N):
        vect_by_quat(array[i], q0, result[i])

def hl_vect_array_by_quats(array, q0, result):
    N = array.shape[0]
    for i in range(N):
        result[i] = hl_vect_by_quat(array[i], q0)



def have_fun():
    # generate code for tests..
    signature = [nb.d[:,:], nb.d[:], nb.d[:,:]]
    fun_numba = jit(arg_types=signature)(vect_array_by_quats_inlined)
    builder = GUFuncVectorize(vect_array_by_quats_inlined, '(m,n), (o) -> (m,n)')
    builder.add(arg_types=signature)
    fun_numbapro = builder.build_ufunc()

    # setup data
    test = np.random.random((16*1024, 3))
    quat = np.random.random((4))
    quat_normalized = np.array(quat.size, dtype = quat.dtype)
    quat_normalize(quat, quat_normalized)
    result = np.zeros_like(test)

    test_args = [ test, quat, result ]
    print_profile_results( profile_functions([
                ('python_f', hl_vect_array_by_quats, test_args),
                ('python_p', vect_array_by_quats, test_args),
                ('numba', fun_numba, test_args),
                ('numbapro', fun_numbapro, test_args),
                ('python_p_inl', vect_array_by_quats_inlined, test_args) 
                ]))

if __name__ == '__main__':
    have_fun()

