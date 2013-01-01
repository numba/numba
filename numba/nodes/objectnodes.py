from numba.nodes import *

class ObjectInjectNode(Node):
    """
    Refer to a Python object in the llvm code.
    """

    _attributes = ['object', 'type']

    def __init__(self, object, type=None, **kwargs):
        super(ObjectInjectNode, self).__init__(**kwargs)
        self.object = object
        self.type = type or object_
        self.variable = Variable(self.type, is_constant=True,
                                 constant_value=object)

    def __repr__(self):
        return "<inject(%s)>" % self.object

NoneNode = ObjectInjectNode(None, object_)

class ObjectTempNode(Node):
    """
    Coerce a node to a temporary which is reference counted.
    """

    _fields = ['node']

    def __init__(self, node, incref=False):
        assert not isinstance(node, ObjectTempNode)
        self.node = node
        self.llvm_temp = None
        self.type = getattr(node, 'type', node.variable.type)
        self.incref = incref

    def __repr__(self):
        return "objtemp(%s)" % self.node

class NoneNode(Node):
    """
    Return None.
    """

    type = typesystem.NoneType()
    variable = Variable(type)

class ObjectTempRefNode(Node):
    """
    Reference an ObjectTempNode, without evaluating its subexpressions.
    The ObjectTempNode must already have been evaluated.
    """

    _fields = []

    def __init__(self, obj_temp_node, **kwargs):
        super(ObjectTempRefNode, self).__init__(**kwargs)
        self.obj_temp_node = obj_temp_node


class IncrefNode(Node):

    _fields = ['value']

    def __init__(self, value, **kwargs):
        super(IncrefNode, self).__init__(**kwargs)
        self.value = value

class DecrefNode(IncrefNode):
    pass