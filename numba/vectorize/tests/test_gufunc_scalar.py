import numpy, ctypes
import numba
import math
from numba.vectorize import GUVectorize

def exp_avg(arr_t, decay_length, out_t):
    decay_factor = math.exp(-1.0 / decay_length)
    sum_w = 0.0
    sum_wx = 0.0
   
    n_t = arr_t.shape[0]
    for t in xrange(n_t):
        sum_w += 1.0
        sum_wx += arr_t[t]
        out_t[t] = sum_wx / sum_w
       
        sum_w *= decay_factor
        sum_wx *= decay_factor

def saxpy(a, X, Y, out):
    for i in range(out.shape[0]):
        out[i] = a * X[i] + Y[i]

def test_exp_avg():
    vect = GUVectorize(exp_avg, '(t),()->(t)')
    vect.add('void(f8[:], f8, f8[:])')
    ufunc = vect.build_ufunc()

    arr_t = numpy.arange(10, dtype=numpy.float64)
    decay_length = numpy.float64(123)
    out_t = numpy.empty(10, dtype=numpy.float64)

    ufunc(arr_t, decay_length, out_t)

    expect = numpy.zeros_like(out_t)
    exp_avg(arr_t, decay_length, expect)

    numpy.allclose(expect, out_t)

def test_saxpy():
    vect = GUVectorize(saxpy, '(),(t),(t)->(t)')
    vect.add('void(f4, f4[:], f4[:], f4[:])')
    gusaxpy = vect.build_ufunc()

    
    A = numpy.array(numpy.float32(2))
    X = numpy.arange(10, dtype=numpy.float32).reshape(5,2)
    Y = numpy.arange(10, dtype=numpy.float32).reshape(5,2)

    out = gusaxpy(A, X, Y)

    for j in range(5):
        for i in range(2):
            exp = A * X[j, i] + Y[j, i]
            assert exp == out[j, i], (exp, out[j, i])

    A = numpy.arange(5, dtype=numpy.float32)
    out = gusaxpy(A, X, Y)

    for j in range(5):
        for i in range(2):
            exp = A[j] * X[j, i] + Y[j, i]
            assert exp == out[j, i]


def main():
    test_exp_avg()
    test_saxpy()

if __name__ == '__main__':
    main()
