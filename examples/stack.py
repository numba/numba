from __future__ import print_function, absolute_import

import gc
from numba.utils import OrderedDict
from numba import njit
from numba.jitclass import jitclass
from numba import deferred_type, intp, optional
from numba.runtime import rtsys

linkednode_spec = OrderedDict()
linkednode_type = deferred_type()
linkednode_spec['data'] = data_type = deferred_type()
linkednode_spec['next'] = optional(linkednode_type)


@jitclass(linkednode_spec)
class LinkedNode(object):
    def __init__(self, data):
        self.data = data
        self.next = None


linkednode_type.define(LinkedNode.class_type.instance_type)

stack_spec = OrderedDict()
stack_spec['head'] = optional(linkednode_type)
stack_spec['size'] = intp


@jitclass(stack_spec)
class Stack(object):
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, data):
        new = LinkedNode(data)
        new.next = self.head
        self.head = new
        self.size += 1

    def pop(self):
        old = self.head
        if old is None:
            raise ValueError("empty")
        else:
            self.head = old.next
            self.size -= 1
            return old.data


data_type.define(intp)


@njit
def test_pushpop(size):
    stack = Stack()

    for i in range(size):
        stack.push(i)

    out = []
    while stack.size > 0:
        out.append(stack.pop())

    return out


def baseline_pushpop(size):
    stack = []
    for i in range(size):
        stack.append(i)

    out = []
    while stack:
        out.append(stack.pop())

    return out


def test_exception():
    stack = Stack()
    stack.push(1)
    assert 1 == stack.pop()
    try:
        # Leaks
        stack.pop()
    except ValueError as e:
        assert 'empty' == str(e)


def runme():
    size = 1000
    test_pushpop(size)
    test_exception()


if __name__ == '__main__':
    runme()
    gc.collect()
    print(rtsys.get_allocation_stats())
