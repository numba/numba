
from __future__ import print_function, absolute_import, division

import collections

from numba import types


QuicksortImplementation = collections.namedtuple(
    'QuicksortImplementation',
    (# The compile function itself
     'compile',
     # All subroutines exercised by test_sort
     'partition', 'partition3', 'insertion_sort',
     # The top-level function
     'run_quicksort',
     ))


Partition = collections.namedtuple('Partition', ('start', 'stop'))

# Under this size, switch to a simple insertion sort
SMALL_QUICKSORT = 15

MAX_STACK = 100


def make_quicksort_impl(wrap):

    intp = types.intp
    zero = intp(0)

    @wrap
    def LT(a, b):
        """
        Trivial comparison function between two keys.  This is factored out to
        make it clear where comparisons occur.
        """
        return a < b

    @wrap
    def insertion_sort(A, low, high):
        """
        Insertion sort A[low:high + 1]. Note the inclusive bounds.
        """
        for i in range(low + 1, high + 1):
            v = A[i]
            # Insert v into A[low:i]
            j = i
            while j > low and v < A[j - 1]:
                # Make place for moving A[i] downwards
                A[j] = A[j - 1]
                j -= 1
            A[j] = v

    @wrap
    def partition(A, low, high):
        """
        Partition A[low:high + 1] around a chosen pivot.  The pivot's index
        is returned.
        """
        mid = (low + high) >> 1
        # NOTE: this pattern of swaps for the pivot choice and the
        # partitioning gives good results (i.e. regular O(n log n))
        # on sorted, reverse-sorted, and uniform arrays.  Subtle changes
        # risk breaking this property.

        # median of three {low, middle, high}
        if A[mid] < A[low]:
            A[low], A[mid] = A[mid], A[low]
        if A[high] < A[mid]:
            A[high], A[mid] = A[mid], A[high]
        if A[mid] < A[low]:
            A[low], A[mid] = A[mid], A[low]
        pivot = A[mid]

        A[high], A[mid] = A[mid], A[high]
        i = low
        j = high - 1
        while True:
            while A[i] < pivot:
                i += 1
            while A[j] > pivot and j >= low:
                j -= 1
            if i >= j:
                break
            A[i], A[j] = A[j], A[i]
            i += 1
            j -= 1
        A[i], A[high] = A[high], A[i]
        return i

    @wrap
    def partition3(A, low, high):
        """
        Three-way partition [low, high) around a chosen pivot.
        A tuple (lt, gt) is returned such that:
            - all elements in [low, lt) are < pivot
            - all elements in [lt, gt] are == pivot
            - all elements in (gt, high] are > pivot
        """
        mid = (low + high) >> 1
        # median of three {low, middle, high}
        if A[mid] < A[low]:
            A[low], A[mid] = A[mid], A[low]
        if A[high] < A[mid]:
            A[high], A[mid] = A[mid], A[high]
        if A[mid] < A[low]:
            A[low], A[mid] = A[mid], A[low]
        pivot = A[mid]

        A[low], A[mid] = A[mid], A[low]
        lt = low
        gt = high
        i = low + 1
        while i <= gt:
            if A[i] < pivot:
                A[lt], A[i] = A[i], A[lt]
                lt += 1
                i += 1
            elif A[i] > pivot:
                A[gt], A[i] = A[i], A[gt]
                gt -= 1
            else:
                i += 1
        return lt, gt

    @wrap
    def run_quicksort(A):
        stack = [Partition(zero, zero)] * 100
        stack[0] = Partition(zero, len(A) - 1)
        n = 1

        while n > 0:
            n -= 1
            low, high = stack[n]
            assert high > low
            # Partition until it becomes more efficient to do an insertion sort
            while high - low >= SMALL_QUICKSORT:
                i = partition(A, low, high)
                # Push largest partition on the stack
                if high - i > i - low:
                    # Right is larger
                    stack[n] = Partition(i + 1, high)
                    n += 1
                    high = i - 1
                else:
                    stack[n] = Partition(low, i - 1)
                    n += 1
                    low = i + 1

            insertion_sort(A, low, high)

    # Unused quicksort implementation based on 3-way partitioning; the
    # partitioning scheme turns out exhibiting bad behaviour on sorted arrays.
    @wrap
    def _run_quicksort(A):
        stack = [Partition(zero, zero)] * 100
        stack[0] = Partition(zero, len(A) - 1)
        n = 1

        while n > 0:
            n -= 1
            low, high = stack[n]
            assert high > low
            # Partition until it becomes more efficient to do an insertion sort
            while high - low >= SMALL_QUICKSORT:
                l, r = partition3(A, low, high)
                # One trivial (empty) partition => iterate on the other
                if r == high:
                    high = l - 1
                elif l == low:
                    low = r + 1
                # Push largest partition on the stack
                elif high - r > l - low:
                    # Right is larger
                    stack[n] = Partition(r + 1, high)
                    n += 1
                    high = l - 1
                else:
                    stack[n] = Partition(low, l - 1)
                    n += 1
                    low = r + 1

            insertion_sort(A, low, high)


    return QuicksortImplementation(wrap,
                                   partition, partition3, insertion_sort,
                                   run_quicksort)


def make_py_quicksort(*args):
    return make_quicksort_impl((lambda f: f), *args)

def make_jit_quicksort(*args):
    from numba import jit
    return make_quicksort_impl((lambda f: jit(nopython=True)(f)),
                               *args)
