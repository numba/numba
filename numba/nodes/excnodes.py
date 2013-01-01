from numba.nodes import *
import numba.nodes

class CheckErrorNode(Node):
    """
    Check for an exception.

        badval: if this value is returned, propagate an error
        goodval: if this value is not returned, propagate an error

    If exc_type, exc_msg and optionally exc_args are given, an error is
    raised instead of propagating it.
    """

    _fields = ['return_value', 'badval', 'raise_node']

    def __init__(self, return_value, badval=None, goodval=None,
                 exc_type=None, exc_msg=None, exc_args=None,
                 **kwargs):
        super(CheckErrorNode, self).__init__(**kwargs)
        self.return_value = return_value

        if badval is not None and not isinstance(badval, ast.AST):
            badval = ConstNode(badval, return_value.type)
        if goodval is not None and not isinstance(goodval, ast.AST):
            goodval = ConstNode(goodval, return_value.type)

        self.badval = badval
        self.goodval = goodval

        self.raise_node = RaiseNode(exc_type, exc_msg, exc_args)

class RaiseNode(Node):

    _fields = ['exc_type', 'exc_msg', 'exc_args']

    def __init__(self, exc_type, exc_msg, exc_args=None, print_on_trap=True,
                 **kwargs):
        super(RaiseNode, self).__init__(**kwargs)
        self.exception_type = None
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            self.exception_type = exc_type
            exc_type = const(exc_type, object_)
        if isinstance(exc_msg, (str, unicode)):
            exc_msg = const(exc_msg, c_string_type)

        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.exc_args = exc_args

        self.print_on_trap = print_on_trap

class PropagateNode(Node):
    """
    Propagate an exception (jump to the error label).
    """
