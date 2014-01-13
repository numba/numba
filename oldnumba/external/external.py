# -*- coding: utf-8 -*-
"""
This module adds a way to declare external functions.

See numba.function_util on how to call them.
"""
from __future__ import print_function, division, absolute_import

import numba

class ExternalFunction(object):
    _attributes = ('func_name', 'arg_types', 'return_type', 'is_vararg',
                   'check_pyerr_occurred', 'badval', 'goodval')
    func_name = None
    arg_types = None
    return_type = None
    is_vararg = False

    badval = None
    goodval = None
    exc_type = None
    exc_msg = None
    exc_args = None

    check_pyerr_occurred = False

    def __init__(self, return_type=None, arg_types=None, **kwargs):
        # Add positional arguments to keyword arguments
        if return_type is not None:
            kwargs['return_type'] = return_type
        if arg_types is not None:
            kwargs['arg_types'] = arg_types

        # Process keyword arguments
        if __debug__:
            # Only accept keyword arguments defined _attributes
            for k, v in kwargs.items():
                if k not in self._attributes:
                    raise TypeError("Invalid keyword arg %s -> %s" % (k, v))
        vars(self).update(kwargs)

    @property
    def signature(self):
        return numba.function(return_type=self.return_type,
                              args=self.arg_types,
                              is_vararg=self.is_vararg)

    @property
    def name(self):
        if self.func_name is None:
            return type(self).__name__
        else:
            return self.func_name

    def declare_lfunc(self, context, llvm_module):
        lfunc_type = self.signature.to_llvm(context)
        lfunc = llvm_module.get_or_insert_function(lfunc_type, name=self.name)
        return lfunc

class ExternalLibrary(object):
    def __init__(self, context):
        self._context = context
        # (name) -> (external function instance)
        self._functions = {}

    def add(self, extfn):
        if __debug__:
            # Sentry for duplicated external function name
            if extfn.name in self._functions:
                raise NameError("Duplicated external function: %s" % extfn.name)
        self._functions[extfn.name] = extfn

    def get(self, name):
        return self._functions[name]

    def __contains__(self, name):
        return name in self._functions

    def declare(self, module, name, arg_types=(), return_type=None):
        extfn = self._functions[name] # raises KeyError if name is not found

        if arg_types and return_type:
            if (extfn.arg_types != arg_types
                and extfn.return_type != return_type):
                raise TypeError("Signature mismatch")

        sig = extfn.signature
        lfunc_type = sig.to_llvm(self._context)
        return sig, module.get_or_insert_function(lfunc_type, extfn.name)



