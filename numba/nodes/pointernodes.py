from numba.nodes import *
import numba.nodes

def pointer_add(pointer, offset):
    assert pointer.type == char.pointer()
    left = numba.nodes.ptrtoint(pointer)
    result = ast.BinOp(left, ast.Add(), offset)
    result.type = left.type
    result.variable = Variable(result.type)
    return CoercionNode(result, char.pointer())

def ptrtoint(node):
    return CoercionNode(node, Py_uintptr_t)

def ptrfromint(intval, dst_ptr_type):
    return CoercionNode(ConstNode(intval, Py_uintptr_t), dst_ptr_type)

def value_at_offset(obj_node, offset, dst_type):
    """
    Perform (dst_type) (((char *) my_object) + offset)
    """
    offset = ConstNode(offset, Py_ssize_t)
    pointer = PointerFromObject(obj_node)
    pointer = CoercionNode(pointer, char.pointer())
    pointer = pointer_add(pointer, offset)
    value_at_offset = CoercionNode(pointer, dst_type)
    return value_at_offset

class DereferenceNode(Node):
    """
    Dereference a pointer
    """

    _fields = ['pointer']

    def __init__(self, pointer, **kwargs):
        super(DereferenceNode, self).__init__(**kwargs)
        self.pointer = pointer
        self.type = pointer.type.base_type

    def __repr__(self):
        return "*%s" % (self.pointer,)

class PointerFromObject(Node):
    """
    Bitcast objects to void *
    """

    _fields = ['node']
    type = void.pointer()
    variable = Variable(type)

    def __init__(self, node, **kwargs):
        super(PointerFromObject, self).__init__(**kwargs)
        self.node = node

    def __repr__(self):
        return "((void *) %s)" % (self.node,)
