
from __future__ import print_function, absolute_import, division

import heapq as hq

import numpy as np

from numba import types
from numba.errors import TypingError
from numba.extending import overload, register_jitable
from numba.numpy_support import as_dtype


@register_jitable
def _siftdown(heap, startpos, pos):
    newitem = heap[pos]

    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break

    heap[pos] = newitem


@register_jitable
def _siftup(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]

    childpos = 2 * pos + 1
    while childpos < endpos:

        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos

        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1

    heap[pos] = newitem
    _siftdown(heap, startpos, pos)


@overload(hq.heapify)
def hq_heapify(x):

    if not isinstance(x, types.List):
        raise TypingError('heap argument must be a list')

    # what to do if list is empty?

    x_dt = as_dtype(x.dtype)
    if np.issubdtype(x_dt, np.complexfloating):
        msg = ("'<' not supported between instances "
               "of 'complex' and 'complex'")
        raise TypingError(msg)

    def hq_heapify_impl(x):
        n = len(x)
        for i in range(n // 2 - 1, -1, -1):
            _siftup(x, i)

    return hq_heapify_impl
