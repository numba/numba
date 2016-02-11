"""
This is a more complicated jitclasses example.
Here, we implement a binarytree and iterative preorder and inorder traversal
function using a handwritten stack.
"""
from __future__ import print_function, absolute_import
import random
from numba.utils import OrderedDict
from numba import njit
from numba import jitclass
from numba import int32, deferred_type, optional
from numba.runtime import rtsys

node_type = deferred_type()

spec = OrderedDict()
spec['data'] = int32
spec['left'] = optional(node_type)
spec['right'] = optional(node_type)


@jitclass(spec)
class TreeNode(object):
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


node_type.define(TreeNode.class_type.instance_type)

stack_type = deferred_type()

spec = OrderedDict()
spec['data'] = TreeNode.class_type.instance_type
spec['next'] = optional(stack_type)


@jitclass(spec)
class Stack(object):
    def __init__(self, data, next):
        self.data = data
        self.next = next


stack_type.define(Stack.class_type.instance_type)


@njit
def push(stack, data):
    return Stack(data, stack)


@njit
def pop(stack):
    return stack.next


@njit
def make_stack(data):
    return push(None, data)


@njit
def list_preorder(node):
    """
    Returns a list of the data by preorder traversing the tree
    """
    out = []

    stack = make_stack(node)

    while stack is not None:
        node = stack.data
        out.append(node.data)
        stack = pop(stack)

        if node.right is not None:
            stack = push(stack, node.right)
        if node.left is not None:
            stack = push(stack, node.left)

    return out


@njit
def list_inorder(node):
    """
    Returns a list of the data by inorder traversing the tree
    """

    out = []

    done = False

    current = node
    stack = None

    while not done:
        if current is not None:
            stack = push(stack, current)
            current = current.left

        else:
            if stack is not None:
                tos = stack.data
                out.append(tos.data)
                stack = pop(stack)
                current = tos.right
            else:
                done = True

    return out


def build_random_tree(size):
    """
    Create a randomly constructred tree that is fairly balanced
    """
    root = TreeNode(0)

    for i in range(1, size):
        cursor = root
        while True:
            choice = random.choice(['L', 'R'])
            if choice == 'L':
                if cursor.left:
                    cursor = cursor.left
                else:
                    cursor.left = TreeNode(i)
                    break
            elif choice == 'R':
                if cursor.right:
                    cursor = cursor.right
                else:
                    cursor.right = TreeNode(i)
                    break
    return root


def build_simple_tree():
    """
    Create a simple tree
    """
    node = TreeNode(1)
    node.left = TreeNode(2)
    node.right = TreeNode(3)
    node.right.left = TreeNode(4)
    node.right.right = TreeNode(5)
    return node


def run(tree):
    preorder = list_preorder(tree)
    print("== Preorder == ")
    print(preorder)

    inorder = list_inorder(tree)
    print("== Inorder == ")
    print(inorder)

    return preorder, inorder


def runme():
    print("== Simple Tree ==")
    preorder, inorder = run(build_simple_tree())
    assert preorder == [1, 2, 3, 4, 5]
    assert inorder == [2, 1, 4, 3, 5]

    print("== Big Random Tree ==")
    run(build_random_tree(100))


if __name__ == '__main__':
    runme()
    print("== Print memory allocation information == ")
    print(rtsys.get_allocation_stats())
