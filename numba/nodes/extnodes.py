from numba.nodes import *

class ExtTypeAttribute(Node):

    _fields = ['value']

    def __init__(self, value, attr, ctx, ext_type, **kwargs):
        super(ExtTypeAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.variable = ext_type.symtab[attr]
        self.type = self.variable.type
        self.ctx = ctx
        self.ext_type = ext_type

    def __repr__(self):
        return "%s.%s" % (self.value, self.attr)

class NewExtObjectNode(Node):
    """
    Instantiate an extension type. Currently unused.
    """

    _fields = ['args']

    def __init__(self, ext_type, args, **kwargs):
        super(NewExtObjectNode, self).__init__(**kwargs)
        self.ext_type = ext_type
        self.args = args

class ExtensionMethod(Node):

    _fields = ['value']
    call_node = None

    def __init__(self, object, attr, **kwargs):
        super(ExtensionMethod, self).__init__(**kwargs)
        ext_type = object.variable.type
        assert ext_type.is_extension
        self.value = object
        self.attr = attr

        method_type, self.vtab_index = ext_type.methoddict[attr]
        self.type = minitypes.FunctionType(return_type=method_type.return_type,
                                           args=method_type.args,
                                           is_bound_method=True)

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
