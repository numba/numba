# -*- coding: utf-8 -*-

"""
Validate method signatures and inheritance compatiblity.
"""

import types
import warnings
import inspect

import numba
from numba import *
from numba import error
from numba import typesystem
from numba.minivect import minitypes


#------------------------------------------------------------------------
# Method Validators
#------------------------------------------------------------------------

class MethodValidator(object):
    "Interface for method validators"

    def validate(self, method, ext_type):
        """
        Validate a Method. Raise an exception for user typing errors.
        """

class ArgcountMethodValidator(MethodValidator):
    """
    Validate a signature against the number of arguments the function expects.
    """

    def validate(self, method, ext_type):
        """
        Validate a signature (which is None if not declared by the user)
        for a method.
        """
        if method.signature is None:
            return

        nargs = method.py_func.__code__.co_argcount - 1 + method.is_static
        if len(method.signature.args) != nargs:
            raise error.NumbaError(
                "Expected %d argument types in function "
                "%s (don't include 'self')" % (nargs, method.name))

class InitMethodValidator(MethodValidator):
    """
    Validate the init method of extension classes.
    """

    def validate(self, method, ext_type):
        if method.name == '__init__' and (method.is_class or method.is_static):
            raise error.NumbaError("__init__ method should not be a class- "
                                   "or staticmethod")

class JitInitMethodValidator(MethodValidator):
    """
    Validate the init method for jit functions. Issue a warning when the
    signature is omitted.
    """

    def validate(self, method, ext_type):
        if method.name == '__init__' and method.signature is None:
            self.check_init_args(method, ext_type)

    def check_init_args(self, method, ext_type):
        if inspect.getargspec(method.py_func).args:
            warnings.warn(
                "Constructor for class '%s' has no signature, "
                "assuming arguments have type 'object'" %
                ext_type.py_class.__name__)


jit_validators = [ArgcountMethodValidator(), InitMethodValidator(), JitInitMethodValidator()]
autojit_validators = [ArgcountMethodValidator(), InitMethodValidator()]

#------------------------------------------------------------------------
# Inheritance Validators
#------------------------------------------------------------------------

class InheritanceValidator(object):
    """
    Interface for validators that check for compatible inheritance trees.
    """

    def validate(self, ext_type, base_ext_type):
        """
        Validate an extension type with its parents.
        """

class AttributeValidator(object):

    def validate(self, ext_type):
        attr_prefix = utils.get_attributes_type(base).is_prefix(struct_type)

        if not attr_prefix or not method_prefix:
            raise error.NumbaError(
                        "Multiple incompatible base classes found: "
                        "%s and %s" % (base, bases[-1]))
