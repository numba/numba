from numba.nodes import *

class ExtTypeAttribute(ExprNode):

    _fields = ['value']

    def __init__(self, value, attr, ctx, ext_type, **kwargs):
        super(ExtTypeAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.type = ext_type.attributedict[attr]
        self.variable = Variable(self.type)
        self.ctx = ctx
        self.ext_type = ext_type

    def __repr__(self):
        return "%s.%s" % (self.value, self.attr)

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
