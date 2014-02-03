# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.nodes import *
import numba.nodes

class CheckErrorNode(ExprNode):
    """
    Check for an exception.

        badval: if this value is returned, propagate an error
        goodval: if this value is not returned, propagate an error

    If exc_type, exc_msg and optionally exc_args are given, an error is
    raised instead of propagating it.

    See RaiseNode for the exc_* arguments.
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

class RaiseNode(ExprNode):
    """
    Raise an exception.

        exception_type: The Python exception type

        exc_type: The Python exception as an AST node
            May be passed in as a Python exception type

        exc_msg: The message to print as an AST node
            May be passed in as a string

        exc_args: If given, must be an list of AST nodes representing the
                  arguments to PyErr_Format (matching the format specifiers
                  at runtime in exc_msg)
    """

    _fields = ['exc_type', 'exc_msg', 'exc_args']

    def __init__(self, exc_type, exc_msg, exc_args=None, print_on_trap=True,
                 **kwargs):
        super(RaiseNode, self).__init__(**kwargs)
        self.exception_type = None
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            self.exception_type = exc_type
            exc_type = const(exc_type, object_)
        if isinstance(exc_msg, (str, unicode)):
            exc_msg = const(exc_msg, char.pointer())

        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.exc_args = exc_args

        self.print_on_trap = print_on_trap

class PropagateNode(ExprNode):
    """
    Propagate an exception (jump to the error label). This is resolved
    at code generation time and can be generated at any moment.
    """

class PyErr_OccurredNode(ExprNode):
    """
    Check for a set Python exception using PyErr_Occurred().

    Can be set any time after type inference. This node is resolved during
    late specialization.
    """

    # TODO: support checking for (value == badval && PyErr_Occurred()) for
    #       efficiency

    _fields = ['node']

    def __init__(self, node):
        self.node = node
        self.variable = node.variable
        self.type = node.type
