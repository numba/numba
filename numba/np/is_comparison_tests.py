import numpy as np
from arrayobj import test_kernel

def test_structured_is():
    a = np.zeros(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    b = a.copy()
    test_kernel(a, b)

def test_cuda_structured_is():
    a = np.zeros(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    b = a.copy()
    cuda.jit(test_kernel)[1, 1](a, b)
    cuda.synchronize()

def test_record_identity():
    a = np.zeros(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    b = a.copy()

    result = test_kernel(a, b)
    assert result == (1, 0, 0, 1), f"Expected (1, 0, 0, 1), got {result}"

def test_cuda_record_identity():
    a = np.zeros(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    b = a.copy()

    result = cuda.jit(test_kernel)[1, 1](a, b)
    cuda.synchronize()

    assert result == (1, 0, 0, 1), f"Expected (1, 0, 0, 1), got {result}"



def test_different_values():
    # Different values, should print all 0's (False)
    a = np.zeros(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    b = np.ones(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    test_kernel(a, b)

def test_mixed_dtype():
    # Different dtypes, should print all 0's (False)
    a = np.zeros(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    b = np.zeros(1, dtype=[("a", "i4"), ("b", "f4")])[0]
    test_kernel(a, b)

def test_equal_values():
    # Equal values, should print (1, 1, 1, 1)
    a = np.ones(1, dtype=[("a", "i4"), ("b", "i4")])[0]
    b = a.copy()
    result = test_kernel(a, b)
    assert result == (1, 1, 1, 1), f"Expected (1, 1, 1, 1), got {result}"

def test_mixed_values():
    # Mixed values, should print (1, 0, 0, 1)
    a = np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")])[0]
    b = np.array([(1, 3)], dtype=[("a", "i4"), ("b", "i4")])[0]
    result = test_kernel(a, b)
    assert result == (1, 0, 0, 1), f"Expected (1, 0, 0, 1), got {result}"

def test_structured_array():
    # Testing structured array, should print (1, 1, 1, 1)
    a = np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")])
    b = a.copy()
    result = test_kernel(a, b)
    assert result == (1, 1, 1, 1), f"Expected (1, 1, 1, 1), got {result}"

# Run the additional tests
test_different_values()
test_mixed_dtype()
test_equal_values()
test_mixed_values()
test_structured_array()
test_structured_is()
test_cuda_structured_is()
test_record_identity()
test_cuda_record_identity()
