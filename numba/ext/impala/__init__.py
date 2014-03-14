from __future__ import print_function, division, absolute_import
from numba.compiler import compile_extra, Flags
from numba import sigutils
from numba.typing.templates import signature
from .target import ImpalaTargetContext
from .typing import impala_typing_context
from .typing import (FunctionContext, AnyVal, BooleanVal, TinyIntVal, SmallIntVal, IntVal,
		     BigIntVal, FloatVal, DoubleVal, StringVal)

def udf(signature):
    def wrapper(pyfunc):
	udfobj = UDF(pyfunc, signature)
	return udfobj
    return wrapper


class UDF(object):
    def __init__(self, pyfunc, signature):
	self.py_func = pyfunc
	self.signature = signature
	self.name = pyfunc.__name__

	args, return_type = sigutils.normalize_signature(signature)
	flags = Flags()
	flags.set('no_compile')
	self._cres = compile_extra(typingctx=impala_typing,
				   targetctx=impala_targets, func=pyfunc,
				   args=args, return_type=return_type,
				   flags=flags, locals={})
	llvm_func = impala_targets.finalize(self._cres.llvm_func, return_type,
					    args)
	self.llvm_func = llvm_func
	self.llvm_module = llvm_func.module


impala_typing = impala_typing_context()
impala_targets = ImpalaTargetContext(impala_typing)
