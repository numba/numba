# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.

import logging
import ast, inspect

from pymothoa.util.descriptor import Descriptor, instanceof
from pymothoa.compiler_errors import FunctionDeclarationError
from module import LLVMModule
from backend import LLVMCodeGenerator
from types import *

logger = logging.getLogger(__name__)

class LLVMFunction(object):
    retty = Descriptor(constant=True)
    argtys = Descriptor(constant=True)

    code_llvm = Descriptor(constant=True)

class LLVMFuncDecl(LLVMFunction):
    def __init__(self, retty, argtys, module, fn_decl):
        self.retty = retty
        self.argtys = argtys
        self.manager = module
        self.code_llvm = fn_decl

class LLVMFuncDef(LLVMFunction):
    code_python = Descriptor(constant=True)

    c_funcptr_type = Descriptor(constant=True)
    c_funcptr = Descriptor(constant=True)

    manager = Descriptor(constant=True)

    def __init__(self, fnobj, retty, argtys, module, fn_decl):
        self.code_python = fnobj
        self.retty = retty
        self.argtys = argtys
        self.manager = module
        self.code_llvm = fn_decl

    def compile(self):
        from pymothoa.compiler_errors import CompilerError, wrap_by_function

        func = self.code_python
        source = inspect.getsource(func)

        logger.debug('Compiling function: %s', func.__name__)

        tree = ast.parse(source)

        assert type(tree).__name__=='Module'
        assert len(tree.body)==1

        # Code generation for LLVM
        try:
            codegen = LLVMCodeGenerator(
                            self.code_llvm,
                            self.retty,
                            self.argtys,
                            symbols=func.func_globals
                        )
            codegen.visit(tree.body[0])
        except CompilerError as e:
            raise wrap_by_function(e, func)

        self.code_llvm.verify()     # verify generated code
        self.manager.jit_engine.optimize_function(self.code_llvm) # optimize generated code to reduce space

        logger.debug('Dump LLVM IR\n%s', self.code_llvm.dump())

    def assembly(self):
        return self.manager.dump_asm(self.code_llvm)

    def prepare_pointer_to_function(self):
        '''Obtain pointer to function from the JIT engine'''
        addr = self.manager.jit_engine.get_pointer_to_function(self.code_llvm)
        # Create binding with ctypes library
        from ctypes import CFUNCTYPE, cast
        c_argtys = map(lambda T: T.ctype(), self.argtys)
        c_retty = self.retty.ctype()
        self.c_funcptr_type = CFUNCTYPE(c_retty, *c_argtys)
        self.c_funcptr = cast( int(addr), self.c_funcptr_type )

    def run_py(self, *args):
        return self.code_python(*args)

    def run_jit(self, *args):
        from itertools import izip
        # Cast the arguments to corresponding types
        argvals = []
        for aty, aval in izip(self.argtys, args):
            argvals.append(aty.argument_adaptor(aval))

        try:
            retval = self.c_funcptr(*argvals)
        except AttributeError: # Has not create binding to the function.
            self.prepare_pointer_to_function()
            # Call the function
            retval = self.c_funcptr(*argvals)

        return retval

    __call__ = run_jit

class LLVMFuncDef_BoolRet(LLVMFuncDef):
    def run_jit(self, *args):
        retval = super(LLVMFuncDef_BoolRet, self).run_jit(*args)
        # workaround for boolean return type
        return bool(retval)

    __call__ = run_jit
