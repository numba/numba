from numba.nodes import *
import numba.nodes

def pointer_add(pointer, offset):
    assert pointer.type == char.pointer()
    left = numba.nodes.ptrtoint(pointer)
    result = ast.BinOp(left, ast.Add(), offset)
    result.type = left.type
    result.variable = Variable(result.type)
    return CoercionNode(result, char.pointer())

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
