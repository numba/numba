from numba.nodes import *

class ExtTypeAttribute(ExprNode):

    _fields = ['value']

    def __init__(self, value, attr, variable, ctx, ext_type, **kwargs):
        super(ExtTypeAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.type = variable.type
        self.variable = variable
        self.ctx = ctx
        self.ext_type = ext_type

    def __repr__(self):
        return "%s.%s" % (self.value, self.attr)

    @classmethod
    def from_known_attribute(cls, value, attr, ctx, ext_type):
        """
        Create an extension type attribute node if the attribute is known
        to exist (and isn't being inferred)
        """
        assert attr in ext_type.attributedict

        import numba.symtab

        variable = numba.symtab.Variable(ext_type.attributedict[attr])
        return cls(value, attr, variable, ctx, ext_type)

class NewExtObjectNode(ExprNode):
    """
    Instantiate an extension type. Currently unused.
    """

    _fields = ['args']

    def __init__(self, ext_type, args, **kwargs):
        super(NewExtObjectNode, self).__init__(**kwargs)
        self.ext_type = ext_type
        self.args = args

class ExtensionMethod(ExprNode):

    _fields = ['value']
    call_node = None

    def __init__(self, object, attr, **kwargs):
        super(ExtensionMethod, self).__init__(**kwargs)
        ext_type = object.variable.type
        assert ext_type.is_extension
        self.value = object
        self.attr = attr

        method = ext_type.methoddict[attr]
        self.type = method.signature
        self.ext_type = ext_type

    def __repr__(self):
        return "%s.%s" % (self.value, self.attr)


#class ExtensionMethodCall(Node):
#    """
#    Low level call that has resolved the virtual method.
#    """
#
#    _fields = ['vmethod', 'args']
#
#    def __init__(self, vmethod, self_obj, args, signature, **kwargs):
#        super(ExtensionMethodCall, self).__init__(**kwargs)
#        self.vmethod = vmethod
#        self.args = args
#        self.signature = signature
#        self.type = signature
