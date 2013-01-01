from numba.nodes import *

#----------------------------------------------------------------------------
# User-extensible nodes
#----------------------------------------------------------------------------

class UserNodeMeta(type):
    def __init__(cls, what, bases=None, dict=None):
        super(UserNodeMeta, cls).__init__(what, bases, dict)
        cls.actual_name = cls.__name__
        cls.__name__ = "UserNode"

    def __repr__(cls):
        return "<class %s>" % cls.actual_name

class UserNode(Node):
    """
    Node that users can subclass and insert in the AST without using mixins
    to provide user-specified functionality.
    """

    __metaclass__ = UserNodeMeta

    _fields = []

    def infer_types(self, type_inferer):
        """
        Infer the type of this node and set it self.type.

        The return value will replace this node in the AST.
        """
        raise NotImplementedError

    def specialize(self, specializer):
        """
        Just before code generation. Useful to rewrite this node in terms
        of other existing fundamental operations.

        Implementing this method is optional.
        """
        specializer.visitchildren(self)
        return self

    def codegen(self, codegen):
        """
        Generate code for this node.

        Must return an LLVM Value.
        """
        raise NotImplementedError

    def __repr__(self):
        return "<%s object>" % self.actual_name

class dont_infer(UserNode):
    """
    Support delayed type inference of the body. E.g. if you want a portion
    <blob> to be inferred elsewhere:

        print x
        <blob>
        print y

    If we want to infer <blob> after the last print, but evaluate it before,
    we can replace these statements with:

        [print x, dont_infer(<blob>), print y, infer_now(<blob>)]
    """

    _fields = ["arg"]

    def __init__(self, arg):
        self.arg = arg

    def infer_types(self, type_inferer):
        return self

    def specialize(self, specializer):
        return specializer.visit(self.arg)

class infer_now(UserNode):
    "See dont_infer above"

    _fields = []

    def __init__(self, arg, dont_infer_node):
        self.arg = arg
        self.dont_infer_node = dont_infer_node

    def infer_types(self, type_inferer):
        self.dont_infer_node.arg = type_inferer.visit(self.arg)
        return None
