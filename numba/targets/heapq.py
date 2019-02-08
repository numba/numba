
# A port of https://github.com/python/cpython/blob/3.7/Lib/heapq.py

from __future__ import print_function, absolute_import, division

import heapq as hq
import numpy as np

from numba import types
from numba.errors import TypingError
from numba.extending import overload, register_jitable


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

    dt = x.dtype
    if isinstance(dt, types.Complex):
        msg = ("'<' not supported between instances "
               "of 'complex' and 'complex'")
        raise TypingError(msg)

    def hq_heapify_impl(x):
        n = len(x)
        for i in range(n // 2 - 1, -1, -1):
            _siftup(x, i)

    return hq_heapify_impl


@register_jitable
def _siftdown_max(heap, startpos, pos):
    newitem = heap[pos]

    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if parent < newitem:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


@register_jitable
def _siftup_max(heap, pos):
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]

    childpos = 2*pos + 1
    while childpos < endpos:

        rightpos = childpos + 1
        if rightpos < endpos and not heap[rightpos] < heap[childpos]:
            childpos = rightpos

        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1

    heap[pos] = newitem
    _siftdown_max(heap, startpos, pos)


@register_jitable
def _heapify_max(x):
    n = len(x)

    for i in range(n // 2 - 1, -1, -1):
        _siftup_max(x, i)


@register_jitable
def _heapreplace_max(heap, item):
    returnitem = heap[0]
    heap[0] = item
    _siftup_max(heap, 0)
    return returnitem


@overload(hq.heappop)
def hq_heappop(heap):

    def hq_heappop_impl(heap):
        lastelt = heap.pop()
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            _siftup(heap, 0)
            return returnitem
        return lastelt

    return hq_heappop_impl


@overload(hq.heappush)
def heappush(heap, item):

    def hq_heappush_impl(heap, item):
        heap.append(item)
        _siftdown(heap, 0, len(heap) - 1)

    return hq_heappush_impl


@overload(hq.heapreplace)
def heapreplace(heap, item):

    def hq_heapreplace(heap, item):
        returnitem = heap[0]
        heap[0] = item
        _siftup(heap, 0)
        return returnitem

    return hq_heapreplace


@overload(hq.nsmallest)
def nsmallest(n, iterable):

    def hq_nsmallest_impl(n, iterable):

        if n == 1:
            out = np.min(np.asarray(iterable))
            return [out]

        size = len(iterable)
        if n >= size:
            return sorted(iterable)[:n]

        it = iter(iterable)
        result = [(elem, i) for i, elem in zip(range(n), it)]

        _heapify_max(result)
        top = result[0][0]
        order = n

        for elem in it:
            if elem < top:
                _heapreplace_max(result, (elem, order))
                top, _order = result[0]
                order += 1
        result.sort()
        return [elem for (elem, order) in result]

    return hq_nsmallest_impl


@overload(hq.nlargest)
def nlargest(n, iterable):

    def hq_nlargest_impl(n, iterable):

        if n == 1:
            out = np.max(np.asarray(iterable))
            return [out]

        size = len(iterable)
        if n >= size:
            return sorted(iterable)[::-1][:n]

        it = iter(iterable)
        result = [(elem, i) for i, elem in zip(range(0, -n, -1), it)]

        hq.heapify(result)
        top = result[0][0]
        order = -n

        for elem in it:
            if top < elem:
                hq.heapreplace(result, (elem, order))
                top, _order = result[0]
                order -= 1
        result.sort(reverse=True)
        return [elem for (elem, order) in result]

    return hq_nlargest_impl
