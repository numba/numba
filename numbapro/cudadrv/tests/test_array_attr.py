import numpy as np
from numbapro import cuda

from .support import testcase, main


@testcase
def test_contigous_2d():
    ary = np.arange(10)
    cary = ary.reshape(2, 5)
    fary = np.asfortranarray(cary)

    dcary = cuda.to_device(cary)
    dfary = cuda.to_device(fary)
    assert dcary.is_c_contigous()
    assert not dfary.is_c_contigous()
    assert not dcary.is_f_contigous()
    assert dfary.is_f_contigous()

@testcase
def test_contigous_3d():
    ary = np.arange(20)
    cary = ary.reshape(2, 5, 2)
    fary = np.asfortranarray(cary)
    
    dcary = cuda.to_device(cary)
    dfary = cuda.to_device(fary)
    assert dcary.is_c_contigous()
    assert not dfary.is_c_contigous()
    assert not dcary.is_f_contigous()
    assert dfary.is_f_contigous()


@testcase
def test_contigous_4d():
    ary = np.arange(60)
    cary = ary.reshape(2, 5, 2, 3)
    fary = np.asfortranarray(cary)

    dcary = cuda.to_device(cary)
    dfary = cuda.to_device(fary)
    assert dcary.is_c_contigous()
    assert not dfary.is_c_contigous()
    assert not dcary.is_f_contigous()
    assert dfary.is_f_contigous()

@testcase
def test_ravel_c():
    ary = np.arange(60)
    reshaped = ary.reshape(2, 5, 2, 3)
    expect = reshaped.ravel(order='C')
    dary = cuda.to_device(reshaped)
    dflat = dary.ravel()
    flat = dflat.copy_to_host()
    assert flat.ndim == 1
    assert np.all(expect == flat)

@testcase
def test_ravel_f():
    ary = np.arange(60)
    reshaped = np.asfortranarray(ary.reshape(2, 5, 2, 3))
    expect = reshaped.ravel(order='F')
    dary = cuda.to_device(reshaped)
    dflat = dary.ravel(order='F')
    flat = dflat.copy_to_host()
    assert flat.ndim == 1
    assert np.all(expect == flat)

@testcase
def test_reshape_c():
    ary = np.arange(10)
    expect = ary.reshape(2, 5)
    dary = cuda.to_device(ary)
    dary_reshaped = dary.reshape(2, 5)
    got = dary_reshaped.copy_to_host()
    assert np.all(expect == got)

@testcase
def test_reshape_f():
    ary = np.arange(10)
    expect = ary.reshape(2, 5, order='F')
    dary = cuda.to_device(ary)
    dary_reshaped = dary.reshape(2, 5, order='F')
    got = dary_reshaped.copy_to_host()
    assert np.all(expect == got)

if __name__ == '__main__':
    main()
