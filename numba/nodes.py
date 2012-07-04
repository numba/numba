from .symtab import Variable

class CoercionNode(object):
    def __init__(self, node, dst_type):
        self.node = node
        self.dst_type = dst_type
        self.variable = Variable(dst_type)

    @classmethod
    def coerce(cls, node_or_nodes, dst_type):
        if isinstance(node_or_nodes, list):
            return [cls(node, dst_type) for node in node_or_nodes]
        return cls(node_or_nodes, dst_type)

class DeferredCoercionNode(CoercionNode):
    """
    Coerce to the type of the given variable. The type of the variable may
    change in the meantime (e.g. may be promoted or demoted).
    """

    def __init__(self, node, variable):
        self.node = node
        self.variable = variable