import numpy as np
from numba import *

def operands(dtype=np.double):
    return np.arange(10, dtype=dtype), np.arange(100, dtype=dtype).reshape(10, 10)

def check_kernel(kernel, *args):
    new_args = [arg.copy() for arg in args]
    result = kernel(*new_args)

    new_args = [arg.copy() for arg in args]
    numpy_result = kernel.py_func(*new_args)

    assert np.allclose(result, numpy_result), numpy_result - result

@autojit
def get_slices(a, b):
    return [
        (a, b),
        (a[:, np.newaxis], b),
        (a[np.newaxis, :, np.newaxis], b),
        (a[np.newaxis, :, np.newaxis], b[np.newaxis, :, :]),
        (a[np.newaxis, :, np.newaxis], b[:, np.newaxis, :]),
        (a[np.newaxis, :, np.newaxis], b[:, :, np.newaxis]),
    ]

@autojit
def broadcast_expr1(m1, m2):
    return m1 + m2

@autojit
def broadcast_expr2(m1, m2):
    m2[...] = m1 + m2
    return m2

@autojit
def broadcast_expr3(m1, m2):
    m2[...] = m1 + m2 - 2
    return m2

@autojit
def broadcast_expr4(m1, m2):
    m2[np.newaxis, :] = m1[np.newaxis, :] + m2[np.newaxis, :]
    return m2

def test(dtype):
    """
    >>> test(np.double)
    >>> test('l')
    >>> test(np.complex128)
    >>> test(np.complex64)

    >> if hasattr(np, 'complex256'):
    ...     test(np.complex256)
    ...
    """
    a, b = operands(dtype)
    views = get_slices(a, b)
    py_views = get_slices.py_func(a, b)

    # test slicing
    for (v1, v2), (v3, v4) in zip(views, py_views):
        assert v1.shape == v3.shape
        assert v2.shape == v4.shape
        assert v1.strides == v3.strides
        assert v2.strides == v4.strides
        assert v1.ctypes.data == v3.ctypes.data
        assert v2.ctypes.data == v4.ctypes.data

        check_kernel(broadcast_expr1, a, b)
        check_kernel(broadcast_expr2, a, b)
        check_kernel(broadcast_expr3, a, b)
        check_kernel(broadcast_expr4, a, b)

@autojit
def broadcast_expr5(m1, m2):
    m2[:, 0] = m1 * m1
    return m2

@autojit
def broadcast_expr6(m1, m2):
    m2[1:-1:2, 0] = m1[1:-1:2] * m1[-2:1:-2]
    return m2

@autojit
def broadcast_expr7(m1, m2):
    m2[1:-1:2, 0, ..., ::2] = (m1[1:-1:2, ..., ::2] *
                               m1[-2:1:-2, ..., ::2])
    return m2

def test_index_slice_assmt(dtype):
    """
    >>> test_index_slice_assmt(np.double)
    >>> test_index_slice_assmt('l')
    >>> test_index_slice_assmt(np.complex64)
    >>> test_index_slice_assmt(np.complex128)
    """
    a, b = operands(dtype)
    check_kernel(broadcast_expr5, a, b)
    check_kernel(broadcast_expr6, a, b)

    b = np.arange(10000).reshape(10, 10, 10, 10)
    a = b[0]
    check_kernel(broadcast_expr7, a, b)

@autojit
def shape_mismatch(a, b):
    b[...] = a + b

@autojit(nopython=True)
def shape_mismatch_nopython(a, b):
    b[...] = a + b

def test_shape_mismatch():
    """
    >>> a, b = operands(np.double)

    >>> shape_mismatch(a[:2], b)
    Traceback (most recent call last):
        ...
    ValueError: ...

    # This will abort, so don't run it :)
    >> shape_mismatch_nopython(a[:2], b)
    ValueError: Shape mismatch while broadcasting
    """

if __name__ == "__main__":
    import numba
    numba.testing.testmod()
