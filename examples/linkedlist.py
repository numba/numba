from __future__ import print_function, absolute_import
from numba.utils import OrderedDict
import numpy as np
from numba import njit
from numba.jitclass import jitclass
from numba import int32, deferred_type, optional
from numba.runtime import rtsys

node_type = deferred_type()

spec = OrderedDict()
spec['data'] = int32
spec['next'] = optional(node_type)


@jitclass(spec)
class LinkedNode(object):
    def __init__(self, data, next):
        self.data = data
        self.next = next

    def prepend(self, data):
        return LinkedNode(data, self)


@njit
def make_linked_node(data):
    return LinkedNode(data, None)


node_type.define(LinkedNode.class_type.instance_type)


@njit
def fill_array(arr):
    head = make_linked_node(0)
    for i in range(1, arr.size):
        head = head.prepend(i)

    c = 0
    while head is not None:
        arr[c] = head.data
        head = head.next
        c += 1


def runme():
    arr = np.zeros(100000, dtype=np.int32)
    fill_array(arr)

    # print(arr)


if __name__ == '__main__':
    runme()
    print(rtsys.get_allocation_stats())
    # Pure python: 104ms
    # jitclass: 2.23s
    # jitclass + make_linked_node: 2.15s
    # jitclass + make_linked_node + fill_array: 34.7ms

    # Pure python: 87.3ms
    # jitclass: 892ms
    # jitclass + make_linked_node: 871ms
    # jitclass + make_linked_node + fill_array: 36.1ms
