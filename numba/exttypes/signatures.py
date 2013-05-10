# -*- coding: utf-8 -*-

"""
Handle signatures of methods in @jit and @autojit classes.
"""

from __future__ import print_function, division, absolute_import

import types

import numba
from numba import *
from numba import error
from numba import typesystem

#------------------------------------------------------------------------
# Parse method signatures
#------------------------------------------------------------------------

class Method(object):
    """
    py_func: the python 'def' function
    """

    def __init__(self, py_func, name, signature, is_class, is_static,
                 nopython=False):
        self.py_func = py_func
        # py_func.live_objects = []

        # Name of this function, py_func.__name__ is inherently unreliable
        self.name = name
        self.signature = signature

        self.is_class = is_class
        self.is_static = is_static

        self.nopython = nopython
        self.template_signature = None

        # Filled out after extension method is compiled
        # (ExtensionCompiler.compile_methods())
        self.wrapper_func = None
        self.lfunc = None
        self.lfunc_pointer = None

    def get_wrapper(self):
        if self.is_class:
            return classmethod(self.wrapper_func)
        elif self.is_static:
            return staticmethod(self.wrapper_func)
        else:
            return self.wrapper_func

    def update_from_env(self, func_env):
        self.lfunc = func_env.lfunc
        self.lfunc_pointer = func_env.translator.lfunc_pointer
        self.wrapper_func = func_env.numba_wrapper_func

    def clone(self):
        return type(self)(self.py_func, self.name, self.signature,
                          self.is_class, self.is_static, self.nopython)


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
# Method Builders
#------------------------------------------------------------------------

class MethodMaker(object):
    """
    Creates Methods from python functions and validates user-declared
    signatures.
    """

    def no_signature(self, method):
        "Called when no signature is found for the method"

    def default_signature(self, method, ext_type):
        """
        Retrieve the default method signature for the given method if
        no user-declared signature exists.
        """
        if has_known_signature(method):
            # We know the argument types, but we don't have a solid
            # infrastucture for inter-procedural type inference yet
            # return typesystem.function(None, [])
            return None
        else:
            return None

    def make_method_type(self, method):
        "Create a method type for the given Method and declared signature"
        restype = method.signature.return_type
        argtypes = method.signature.args
        signature = typesystem.ExtMethodType(
                    return_type=restype, args=argtypes, name=method.name,
                    is_class_method=method.is_class, is_static_method=method.is_static)
        return signature

def has_known_signature(method):
    argcount = method.py_func.__code__.co_argcount
    return ((method.is_static and argcount == 0) or
            (not method.is_static and argcount == 1))

# ______________________________________________________________________
# Method processing for @jit classes

class JitMethodMaker(MethodMaker):

    def no_signature(self, py_func):
        if py_func.__name__ != '__init__':
            raise error.NumbaError(
                "Method '%s' does not have signature" % (py_func.__name__,))

    def default_signature(self, method, ext_type):
        if method.name == '__init__':
            argtypes = [numba.object_] * (method.py_func.__code__.co_argcount - 1)
            default_signature = numba.void(*argtypes)
            return default_signature
        else:
            return super(JitMethodMaker, self).default_signature(
                method, ext_type)

# ______________________________________________________________________
# Method processing for @autojit classes

class AutojitMethodMaker(MethodMaker):

    def __init__(self, argtypes):
        self.argtypes = argtypes

    def default_signature(self, method, ext_type):
        if method.name == '__init__':
            default_signature = numba.void(*self.argtypes)
            return default_signature
        else:
            return super(AutojitMethodMaker, self).default_signature(
                method, ext_type)



#------------------------------------------------------------------------
# Method signature parsing
#------------------------------------------------------------------------

def method_argtypes(method, ext_type, argtypes):
    if method.is_static:
        leading_arg_types = ()
    elif method.is_class:
        leading_arg_types = (numba.object_,)
    else:
        leading_arg_types = (ext_type,)

    return leading_arg_types + tuple(argtypes)

def process_signature(method, method_name, method_maker=MethodMaker()):
    """
    Verify a method signature.

    Returns a Method object and the resolved signature.
    Returns None if the object isn't a method.
    """

    signature = None

    is_static=False
    is_class=False

    while True:
        if isinstance(method, types.FunctionType):
            # Process function
            if signature is None:
                method_maker.no_signature(method)

            method = Method(method, method_name, signature,
                            is_class, is_static)
            return method

        elif isinstance(method, typesystem.Function):
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

    assert False # Unreachable

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

    def update_signature(self, method):
        """
        Update a method signature with the extension type for 'self'.

            class Foo(object):
                @void()                 # becomes: void(ext_type(Foo))
                def method(self): ...
        """
        argtypes = method_argtypes(method, self.ext_type, method.signature.args)
        restype = method.signature.return_type
        method.signature = typesystem.function(restype, argtypes)

        method.signature = self.method_maker.make_method_type(method)

    def get_method_signatures(self):
        """
        Return [Method] for each decorated method in the class
        """
        methods = []

        for method_name, method in sorted(self.class_dict.iteritems()):
            method = process_signature(method, method_name)
            if method is None:
                continue

            for validator in self.validators:
                validator.validate(method, self.ext_type)

            if method.signature is None:
                method.signature = self.method_maker.default_signature(
                    method, self.ext_type)

            if method.signature is not None:
                self.update_signature(method)

            methods.append(method)

        return methods

