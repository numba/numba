# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.
#
#

import types

class JITModule(object):
    def __init__(self, name, modargs={}):
        from llvm_backend.module import LLVMModule
        self.module = LLVMModule(name, **modargs)

    def function(self, func=None, ret=types.Void, args=[], later=False):
        def wrapper(func):
            assert type(func).__name__=='function', (
                    '"%s" is not a function.'%func.__name__
            )
            llvmfn = self.module.new_function(func, ret, args)

            if not later: # compile later flag
                llvmfn.compile()

            return llvmfn

        if func is None:
            return wrapper
        else:
            return wrapper(func)

    def declaration(self, ret=types.Void, args=[]):
        def wrapper(func):
            namespace = func.func_globals['__name__']
            realname = '.'.join([namespace, func.__name__])
            return self.module.new_declaration(realname, ret, args)
        return wrapper

    def declare_builtin(self, name, ret=types.Void, args=[]):
        return self.module.new_declaration(name, ret, args)

    def optimize(self):
        self.module.optimize()
        self.module.verify()

    def __str__(self):
        return self.module.dump()

    __repr__ = __str__

default_module = JITModule('default')

function = default_module.function
declare_builtin = default_module.declare_builtin
declaration = default_module.declaration

