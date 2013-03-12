"""
Handle signatures of methods in @jit and @autojit classes.
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
# Parse method signatures
#------------------------------------------------------------------------

class Method(object):
    """
    py_func: the python 'def' function
    """

    def __init__(self, py_func, name, signature, is_class, is_static):
        self.py_func = py_func
        # Name of this function, py_func.__name__ is inherently unreliable
        self.name = name
        self.signature = signature
        py_func.live_objects = []
        self.is_class = is_class
        self.is_static = is_static

    def result(self, py_func):
        if self.is_class:
            return classmethod(py_func)
        elif self.is_static:
            return staticmethod(py_func)
        else:
            return py_func

#------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------

def get_classmethod_func(func):
    """
    Get the Python function the classmethod or staticmethod is wrapping.

    In Python2.6 classmethod and staticmethod don't have the '__func__'
    attribute.
    """
    if isinstance(func, classmethod):
        return func.__get__(object()).__func__
    else:
        assert isinstance(func, staticmethod)
        return func.__get__(object())

#------------------------------------------------------------------------
# Method Validators
#------------------------------------------------------------------------

class Validator(object):
    "Interface for method validators"

    def validate(self, method, ext_type):
        """
        Validate a Method. Raise an exception for user typing errors.
        """

class ArgcountValidator(Validator):
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

class InitValidator(Validator):
    """
    Validate the init method of extension classes.
    """

    def validate(self, method, ext_type):
        if method.name == '__init__' and (method.is_class or method.is_static):
            raise error.NumbaError("__init__ method should not be a class- "
                                   "or staticmethod")

class JitInitValidator(Validator):
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


jit_validators = [ArgcountValidator(), InitValidator(), JitInitValidator()]
autojit_validators = [ArgcountValidator(), InitValidator()]

#------------------------------------------------------------------------
# Method Validators
#------------------------------------------------------------------------

class MethodMaker(object):
    """
    Creates Methods from python functions and validates user-declared
    signatures.
    """

    def __init__(self, ext_type):
        self.ext_type = ext_type

    def no_signature(self, method):
        "Called when no signature is found for the method"

    def default_signature(self, method):
        """
        Retrieve the default method signature for the given method if
        no user-declared signature exists.
        """

    def make_method_type(self, method):
        "Create a method type for the given Method and declared signature"
        restype = method.signature.return_type
        argtypes = method.signature.args
        signature = typesystem.ExtMethodType(
                    return_type=restype, args=argtypes, name=method.name,
                    is_class=method.is_class, is_static=method.is_static)
        return signature

# ______________________________________________________________________
# Method processing for @jit classes

class JitMethodMaker(MethodMaker):

    def no_signature(self, py_func):
        if py_func.__name__ != '__init__':
            raise error.NumbaError(
                "Method '%s' does not have signature" % (py_func.__name__,))

    def default_signature(self, method):
        if method.name == '__init__' and method.signature is None:
            argtypes = [numba.object_] * (method.py_func.__code__.co_argcount - 1)
            default_signature = numba.void(*argtypes)
            return default_signature
        else:
            return None

# ______________________________________________________________________
# Method processing for @autojit classes

class AutojitMethodMaker(MethodMaker):

    def __init__(self, ext_type, argtypes):
        super(AutojitMethodMaker, self).__init__(ext_type)
        self.argtypes = argtypes

    def default_signature(self, method):
        if method.name == '__init__' and method.signature is None:
            default_signature = numba.void(*self.argtypes)
            return default_signature
        else:
            return None


#------------------------------------------------------------------------
# Method signature parsing
#------------------------------------------------------------------------

class MethodSignatureProcessor(object):
    """
    Processes signatures of extension types.
    """

    def __init__(self, class_dict, ext_type, method_maker, validators):
        self.class_dict = class_dict
        self.ext_type = ext_type
        self.method_maker = method_maker

        # List of method validators: [MethodValidator]
        self.validators = validators

    def process_signature(self, method, method_name,
                          is_static=False, is_class=False):
        """
        Verify a method signature.

        Returns a Method object and the resolved signature.
        Returns None if the object isn't a method.
        """

        signature = None

        while True:
            if isinstance(method, types.FunctionType):
                # Process function
                if signature is None:
                    self.method_maker.no_signature(method)

                method = Method(method, method_name, signature,
                                is_class, is_static)
                return method

            elif isinstance(method, minitypes.Function):
                # @double(...)
                # def func(self, ...): ...
                signature = method.signature
                method = method.py_func

            else:
                # Process staticmethod and classmethod
                if isinstance(method, staticmethod):
                    is_static = True
                elif isinstance(method, classmethod):
                    is_class = True
                else:
                    return None

                method = get_classmethod_func(method)

    def update_signature(self, method):
        """
        Update a method signature with the extension type for 'self'.

            class Foo(object):
                @void()                 # becomes: void(ext_type(Foo))
                def method(self): ...
        """
        if method.is_static:
            leading_arg_types = ()
        elif method.is_class:
            leading_arg_types = (numba.object_,)
        else:
            leading_arg_types = (self.ext_type,)

        argtypes = leading_arg_types + method.signature.args
        restype = method.signature.return_type
        method.signature = restype(*argtypes)

    def get_method_signatures(self):
        """
        Return ([Method], [ExtMethodType]) for each decorated method in the class
        """
        methods = []
        method_types = []

        for method_name, method in self.class_dict.iteritems():
            method = self.process_signature(method, method_name)
            if method is None:
                continue

            for validator in self.validators:
                validator.validate(method, self.ext_type)

            if method.signature is None:
                method.signature = self.method_maker.default_signature(method)

            self.update_signature(method)

            method_type = self.method_maker.make_method_type(method)

            methods.append(method)
            method_types.append(method_type)

        return methods, method_types

