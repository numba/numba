"""
This example demonstrates jitclasses and deferred types for writing a
singly-linked-list.
"""
from __future__ import print_function, absolute_import
from collections import OrderedDict
import numpy as np
from numba import njit
from numba import jitclass
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
    """
    Fills the array with n, n - 1, n - 2 and so on
    First we populate a linked list with values 1 ... n
    Then, we traverse the the linked list in reverse and put the value
    into the array from the index.
    """
    head = make_linked_node(0)
    for i in range(1, arr.size):
        head = head.prepend(i)

    c = 0
    while head is not None:
        arr[c] = head.data
        head = head.next
        c += 1


def runme():
    arr = np.zeros(10, dtype=np.int32)
    fill_array(arr)
    print("== Result ==")
    print(arr)
    # Check answer
    np.testing.assert_equal(arr, np.arange(arr.size, dtype=arr.dtype)[::-1])


if __name__ == '__main__':
    runme()
    print("== Print memory allocation information == ")
    print(rtsys.get_allocation_stats())
