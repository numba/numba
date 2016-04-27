"""
Implementation of compiled C callbacks (@cfunc).
"""

from __future__ import print_function, division, absolute_import

import ctypes

from llvmlite import ir

from . import config, sigutils, utils
from .dispatcher import _FunctionCompiler
from .targets import registry
from .typing import signature
from .typing.ctypes_utils import to_ctypes


class _CFuncCompiler(_FunctionCompiler):

    def _customize_flags(self, flags):
        flags.set('no_cpython_wrapper', True)
        flags.set('no_compile', True)
        return flags


class CFunc(object):
    _targetdescr = registry.cpu_target

    def __init__(self, pyfunc, sig, locals, options):
        args, return_type = sig
        if return_type is None:
            raise TypeError("C callback needs an explicit return type")
        self._pyfunc = pyfunc
        self._sig = signature(return_type, *args)
        self._compiler = _CFuncCompiler(pyfunc, self._targetdescr,
                                        options, locals)

    def compile(self):
        self._do_compile()

    def _do_compile(self):
        sig = self._sig

        # Compile native function
        cres = self._compiler.compile(sig.args, sig.return_type)
        if cres.objectmode:
            raise RuntimeError("object mode not allowed in cfuncs")
        fndesc = cres.fndesc

        # Create a separate LLVM module for the C wrapper
        #library = cres.library.codegen.create_library("cfunc")
        library = cres.library
        module = library.create_ir_module(fndesc.unique_name)

        # Compile C wrapper
        #typing_context = self._targetdescr.typing_context
        context = cres.target_context
        ll_argtypes = [context.get_value_type(ty) for ty in sig.args]
        ll_return_type = context.get_value_type(sig.return_type)

        wrapty = ir.FunctionType(ll_return_type, ll_argtypes)
        wrapfn = module.add_function(wrapty, fndesc.llvm_cfunc_wrapper_name)
        builder = ir.IRBuilder(wrapfn.append_basic_block('entry'))

        self._build_c_wrapper(context, builder, cres, wrapfn.args)

        library.add_ir_module(module)
        library.finalize()

        # Keep compile result alive
        self._library = library
        self._wrapper_name = wrapfn.name
        self._wrapper_address = library.get_pointer_to_function(self._wrapper_name)

    def _build_c_wrapper(self, context, builder, cres, c_args):
        sig = self._sig

        fnty = context.call_conv.get_function_type(sig.return_type, sig.args)
        fn = builder.module.add_function(fnty, cres.fndesc.llvm_func_name)

        # XXX no obvious way to freeze an environment
        status, out = context.call_conv.call_function(
            builder, fn, sig.return_type, sig.args, c_args, env=None)

        # XXX what to do with the status?
        builder.ret(out)

    @property
    def native_name(self):
        """
        """
        return self._wrapper_name

    @property
    def address(self):
        """
        """
        return self._wrapper_address

    @utils.cached_property
    def ctypes(self):
        """
        """
        ctypes_args = [to_ctypes(ty) for ty in self._sig.args]
        ctypes_restype = to_ctypes(self._sig.return_type)
        functype = ctypes.CFUNCTYPE(ctypes_restype, *ctypes_args)
        return functype(self.address)
