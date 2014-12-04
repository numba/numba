import numba

@numba.njit("void(f8[:], i4[:], i4, i4)")
def _dual_swap(values, indices, i1, i2):
    dtmp = values[i1]
    values[i1] = values[i2]
    values[i2] = dtmp

    itmp = indices[i1]
    indices[i1] = indices[i2]
    indices[i2] = itmp

@numba.njit("i4(f8[:], i4[:], i4, i4)")
def _partition(values, indices, start, end):
    pivot_idx = start + (end - start + 1) / 2
    _dual_swap(values, indices, start, pivot_idx)
    pivot = values[start]
    i = start + 1
    j = start + 1

    while j <= end:
        if values[j] <= pivot:
            _dual_swap(values, indices, i, j)
            i += 1
        j += 1

    _dual_swap(values, indices, start, i - 1)

    return i - 1

@numba.njit("void(f8[:], i4[:], i4, i4)")
def quicksort(values, indices, start, end):
    size = end - start + 1

    if size > 1:
        i = _partition(values, indices, start, end)
        quicksort(values, indices, start, i - 1)
        quicksort(values, indices, i + 1, end)

if __name__ == '__main__':
    import numpy as np
    from numpy.testing import assert_array_equal

    rng = np.random.RandomState(0)
    values = rng.rand(500)
    indices = np.arange(len(values)).astype(np.int32)

    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_indices = indices[sorted_idx]

    quicksort(values, indices, 0, len(values) - 1)

    assert_array_equal(sorted_values, values)
    assert_array_equal(sorted_indices, indices)
