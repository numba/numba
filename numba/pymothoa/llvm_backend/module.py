# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.
#
# NOTE: LLVM does not permit inter module call. Either the caller module is
# linked with the callee module. Or, a function pointer is brought from the callee
# to the caller. (Any alternative?)
#
import logging
logger = logging.getLogger()

from pymothoa import types
from pymothoa.util.descriptor import Descriptor, instanceof
from types import LLVMType
import llvm # binding

class LLVMModule(object):
    jit_engine = Descriptor(constant=True)

    def __init__(self, name, optlevel=3, vectorize=True):
        self.jit_engine = llvm.JITEngine(name, optlevel, vectorize)

    def optimize(self):
        self.jit_engine.optimize()

    def verify(self):
        self.jit_engine.verify()

    def dump_asm(self, fn):
        return self.jit_engine.dump_asm(fn)

    def dump(self):
        return self.jit_engine.dump()

    def _new_func_def_or_decl(self, ret, args, name_or_func):
        from function import LLVMFuncDef, LLVMFuncDecl, LLVMFuncDef_BoolRet
        is_func_def = not isinstance(name_or_func, basestring)
        if is_func_def:
            func = name_or_func
            namespace = func.func_globals['__name__']
            realname = '.'.join([namespace, func.__name__])
        else:
            name = name_or_func
            realname = name

        # workaround for boolean return type
        is_ret_bool = False
        if ret is types.Bool:
            # Change return type to 8-bit int
            retty = LLVMType(types.Int8)
            is_ret_bool = True
            logger.warning('Using workaround (change to Int8) for boolean return type.')
        else:
            retty = LLVMType(ret)

        # workaround for boolean argument type
        argtys = []
        count_converted_boolean = 0
        for arg in args:
            if arg is types.Bool:
                argtys.append(LLVMType(types.Int8))
                count_converted_boolean += 1
            else:
                argtys.append(LLVMType(arg))
        else:
            if count_converted_boolean:
                logger.warning('Using workaround (changed to Int8) for boolean argument type.')

        fn_decl = self.jit_engine.make_function(
                    realname,
                    retty.type(),
                    map(lambda X: X.type(), argtys),
                  )

        if fn_decl.name() != realname:
            raise NameError(
                    'Generated function has a different name: %s'%(
                        fn_decl.name()))

        if is_func_def:
            if is_ret_bool:
                return LLVMFuncDef_BoolRet(func, retty, argtys, self, fn_decl)
            else:
                return LLVMFuncDef(func, retty, argtys, self, fn_decl)
        else:
            return LLVMFuncDecl(retty, argtys, self, fn_decl)

    def new_function(self, func, ret, args):
        return self._new_func_def_or_decl(ret, args, func)

    def new_declaration(self, realname, ret, args):
        return self._new_func_def_or_decl(ret, args, realname)

