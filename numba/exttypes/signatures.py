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

def validate_method(py_func, sig, is_static):
    assert isinstance(py_func, types.FunctionType)

    nargs = py_func.__code__.co_argcount - 1 + is_static
    if len(sig.args) != nargs:
        raise error.NumbaError(
            "Expected %d argument types in function "
            "%s (don't include 'self')" % (nargs, py_func.__name__))

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

class MethodMaker(object):
    """
    Creates Methods from python functions and validates user-declared
    signatures.
    """

    def __init__(self, ext_type):
        self.ext_type = ext_type

    def no_signature(self, method):
        "Called when no signature is found for the method"

    def default_signature(self, method, method_name):
        "Retrieve the default method signature for the given method"

    def make_method_type(self, method):
        "Create a method type for the given Method and declared signature"

# ______________________________________________________________________
# Method processing for @jit classes

class JitMethodMaker(MethodMaker):

    def no_signature(self, method):
        raise error.NumbaError(
            "Method '%s' does not have signature" % (method.__name__,))

    def validate_init_method(self, init_method):
        if inspect.getargspec(init_method).args:
            warnings.warn(
                "Constructor for class '%s' has no signature, "
                "assuming arguments have type 'object'" %
                self.ext_type.py_class.__name__)

    def default_signature(self, method, method_name):
        if (method_name == '__init__' and
                isinstance(method, types.FunctionType)):
            self.validate_init_method(method)

            argtypes = [numba.object_] * (method.__code__.co_argcount - 1)
            default_signature = numba.void(*argtypes)
            return default_signature
        else:
            return None

    def make_method_type(self, method):
        restype = method.signature.return_type
        argtypes = method.signature.args
        signature = typesystem.ExtMethodType(
                    return_type=restype, args=argtypes, name=method.name,
                    is_class=method.is_class, is_static=method.is_static)
        return signature

#------------------------------------------------------------------------
# Method signature parsing
#------------------------------------------------------------------------

class MethodSignatureProcessor(object):
    """
    Processes signatures of extension types.
    """

    def __init__(self, class_dict, ext_type, method_maker):
        self.class_dict = class_dict
        self.ext_type = ext_type
        self.method_maker = method_maker

    def get_signature(self, is_class, is_static, sig):
        """
        Create a signature given the user-specified signature. E.g.

            class Foo(object):
                @void()                 # becomes: void(ext_type(Foo))
                def method(self): ...
        """
        if is_static:
            leading_arg_types = ()
        elif is_class:
            leading_arg_types = (numba.object_,)
        else:
            leading_arg_types = (self.ext_type,)

        argtypes = leading_arg_types + sig.args
        restype = sig.return_type
        return minitypes.FunctionType(return_type=restype, args=argtypes)

    def process_signature(self, method, method_name, default_signature,
                          is_static=False, is_class=False):
        """
        Verify a method signature.

        Returns a Method object and the resolved signature.
        """
        while True:
            if isinstance(method, types.FunctionType):
                # Process function
                if default_signature is None:
                    self.method_maker.no_signature(method)

                validate_method(method, default_signature or object_(),
                                is_static)
                if default_signature is None:
                    default_signature = minitypes.FunctionType(return_type=None,
                                                               args=[])
                sig = self.get_signature(is_class, is_static, default_signature)
                method = Method(method, method_name, sig, is_class, is_static)
                return method

            elif isinstance(method, minitypes.Function):
                # @double(...)
                # def func(self, ...): ...
                default_signature = method.signature
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

    def get_method_signatures(self):
        """
        Return [Method] for each decorated method in the class
        """
        methods = []

        for method_name, method in self.class_dict.iteritems():
            default_signature = self.method_maker.default_signature(method,
                                                                    method_name)

            method = self.process_signature(method, method_name,
                                            default_signature)
            if method is None:
                continue

            method_type = self.method_maker.make_method_type(method)
            methods.append((method, method_type))

        return methods

