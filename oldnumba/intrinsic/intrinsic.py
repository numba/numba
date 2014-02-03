# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core
from numba.typesystem import function

from collections import namedtuple

Signature = namedtuple('Signature', ['return_type', 'arg_types'])

class Intrinsic(object):
    _attributes = ('func_name', 'arg_types', 'return_type')
    func_name = None
    arg_types = None
    return_type = None
    linkage = llvm.core.LINKAGE_LINKONCE_ODR
    is_vararg = False

    # Unused?
    #    badval = None
    #    goodval = None
    #    exc_type = None
    #    exc_msg = None
    #    exc_args = None

    def __init__(self, **kwargs):
        if __debug__:
            # Only accept keyword arguments defined _attributes
            for k, v in kwargs.items():
                if k not in self._attributes:
                    raise TypeError("Invalid keyword arg %s -> %s" % (k, v))
        vars(self).update(kwargs)

    @property
    def signature(self):
        return function(self.return_type, self.arg_types, self.is_vararg)

    @property
    def name(self):
        if self.func_name is None:
            return type(self).__name__
        else:
            return self.func_name

    def implementation(self, module, lfunc):
        return None


class IntrinsicLibrary(object):
    '''An intrinsic library maintains a LLVM module for holding the
        intrinsics.  These are functions are used internally to implement
        specific features.
        '''
    def __init__(self, context):
        self._context = context
        self._module = llvm.core.Module.new('intrlib.%X' % id(self))
        # A lookup dictionary that matches (name) -> (intr)
        self._functions = {}
        # (name, args) -> (lfunc)
        self._compiled = {}
        # Build function pass manager to reduce memory usage of
        from llvm.passes import FunctionPassManager, PassManagerBuilder
        pmb = PassManagerBuilder.new()
        pmb.opt_level = 2
        self._fpm = FunctionPassManager.new(self._module)
        pmb.populate(self._fpm)

    def add(self, intr):
        '''Add a new intrinsic.
        intr --- an Intrinsic class
        '''
        if __debug__:
            # Sentry for duplicated external function name
            if intr.__name__ in self._functions:
                raise NameError("Duplicated intrinsic function: %s" \
                                % intr.__name__)
        self._functions[intr.__name__] = intr
        if intr.arg_types and intr.return_type:
            # only if it defines arg_types and return_type
            self.implement(intr())

    def implement(self, intr):
        '''Implement a new intrinsic.
        intr --- an Intrinsic class
        '''
        # implement the intrinsic
        lfunc_type = intr.signature.to_llvm(self._context)
        lfunc = self._module.add_function(lfunc_type, name=intr.name)
        lfunc.linkage = intr.linkage
        intr.implementation(lfunc.module, lfunc)

        # optimize the function
        self._fpm.run(lfunc)

        # populate the lookup dictionary
        key = type(intr).__name__, tuple(intr.arg_types)
        sig = intr.signature.to_llvm(self._context),
        self._compiled[key] = sig, lfunc

    def declare(self, module, name, arg_types=(), return_type=None):
        '''Create a declaration in the module.
        '''
        sig, intrlfunc = self.get(name, arg_types, return_type)
        lfunc = module.get_or_insert_function(intrlfunc.type.pointee,
                                              name=intrlfunc.name)
        return sig, lfunc
    
    def get(self, name, arg_types=(), return_type=None):
        '''Get an intrinsic by name and sig

        name --- function name
        arg_types --- function arg types
        return_types --- function return type
            
        Returns the function signature and a lfunc pointing to the 
        '''
        if not arg_types and not return_type:
            intr = self._functions[name]
            sig, lfunc = self.get(name,
                                  arg_types=intr.arg_types,
                                  return_type=intr.return_type)
        else:
            key = name, tuple(arg_types)
            try:
                sig, lfunc = self._compiled[key]
            except KeyError:
                intr = self._functions[name]
                if not intr.arg_types and not intr.return_type:
                    self.implement(intr(arg_types=arg_types,
                                        return_type=return_type))
                sig, lfunc = self._compiled[key]

        return sig, lfunc

    def link(self, module):
        '''Link the intrinsic library into the target module.
        '''
        module.link_in(self._module, preserve=True)
