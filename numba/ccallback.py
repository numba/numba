"""
Implementation of compiled C callbacks (@cfunc).
"""

from __future__ import print_function, division, absolute_import

import ctypes

from llvmlite import ir

from . import utils, compiler
from .caching import NullCache, FunctionCache
from .dispatcher import _FunctionCompiler
from .targets import registry
from .typing import signature
from .typing.ctypes_utils import to_ctypes


class _CFuncCompiler(_FunctionCompiler):

    def _customize_flags(self, flags):
        flags.set('no_cpython_wrapper', True)
        # Disable compilation of the IR module, because we first want to
        # add the cfunc wrapper.
        flags.set('no_compile', True)
        # Object mode is not currently supported in C callbacks
        # (no reliable way to get the environment)
        flags.set('enable_pyobject', False)
        if flags.force_pyobject:
            raise NotImplementedError("object mode not allowed in C callbacks")
        return flags


class CFunc(object):
    """
    A compiled C callback, as created by the @cfunc decorator.
    """
    _targetdescr = registry.cpu_target

    def __init__(self, pyfunc, sig, locals, options,
                 pipeline_class=compiler.Pipeline):
        args, return_type = sig
        if return_type is None:
            raise TypeError("C callback needs an explicit return type")
        self.__name__ = pyfunc.__name__
        self.__qualname__ = getattr(pyfunc, '__qualname__', self.__name__)
        self.__wrapped__ = pyfunc

        self._pyfunc = pyfunc
        self._sig = signature(return_type, *args)
        self._compiler = _CFuncCompiler(pyfunc, self._targetdescr,
                                        options, locals,
                                        pipeline_class=pipeline_class)

        self._wrapper_name = None
        self._wrapper_address = None
        self._cache = NullCache()
        self._cache_hits = 0

    def enable_caching(self):
        self._cache = FunctionCache(self._pyfunc)

    @compiler.global_compiler_lock
    def compile(self):
        # Try to load from cache
        cres = self._cache.load_overload(self._sig, self._targetdescr.target_context)
        if cres is None:
            cres = self._compile_uncached()
            self._cache.save_overload(self._sig, cres)
        else:
            self._cache_hits += 1

        self._library = cres.library
        self._wrapper_name = cres.fndesc.llvm_cfunc_wrapper_name
        self._wrapper_address = self._library.get_pointer_to_function(self._wrapper_name)

    def _compile_uncached(self):
        sig = self._sig

        # Compile native function
        cres = self._compiler.compile(sig.args, sig.return_type)
        assert not cres.objectmode  # disabled by compiler above
        fndesc = cres.fndesc

        # Compile C wrapper
        # Note we reuse the same library to allow inlining the Numba
        # function inside the wrapper.
        library = cres.library
        module = library.create_ir_module(fndesc.unique_name)
        context = cres.target_context
        ll_argtypes = [context.get_value_type(ty) for ty in sig.args]
        ll_return_type = context.get_value_type(sig.return_type)

        wrapty = ir.FunctionType(ll_return_type, ll_argtypes)
        wrapfn = module.add_function(wrapty, fndesc.llvm_cfunc_wrapper_name)
        builder = ir.IRBuilder(wrapfn.append_basic_block('entry'))

        self._build_c_wrapper(context, builder, cres, wrapfn.args)

        library.add_ir_module(module)
        library.finalize()

        return cres

    def _build_c_wrapper(self, context, builder, cres, c_args):
        sig = self._sig
        pyapi = context.get_python_api(builder)

        fnty = context.call_conv.get_function_type(sig.return_type, sig.args)
        fn = builder.module.add_function(fnty, cres.fndesc.llvm_func_name)

        # XXX no obvious way to freeze an environment
        status, out = context.call_conv.call_function(
            builder, fn, sig.return_type, sig.args, c_args)

        with builder.if_then(status.is_error, likely=False):
            # If (and only if) an error occurred, acquire the GIL
            # and use the interpreter to write out the exception.
            gil_state = pyapi.gil_ensure()
            context.call_conv.raise_error(builder, pyapi, status)
            cstr = context.insert_const_string(builder.module, repr(self))
            strobj = pyapi.string_from_string(cstr)
            pyapi.err_write_unraisable(strobj)
            pyapi.decref(strobj)
            pyapi.gil_release(gil_state)

        builder.ret(out)

    @property
    def native_name(self):
        """
        The process-wide symbol the C callback is exposed as.
        """
        # Note from our point of view, the C callback is the wrapper around
        # the native function.
        return self._wrapper_name

    @property
    def address(self):
        """
        The address of the C callback.
        """
        return self._wrapper_address

    @utils.cached_property
    def cffi(self):
        """
        A cffi function pointer representing the C callback.
        """
        import cffi
        ffi = cffi.FFI()
        # cffi compares types by name, so using precise types would risk
        # spurious mismatches (such as "int32_t" vs. "int").
        return ffi.cast("void *", self.address)

    @utils.cached_property
    def ctypes(self):
        """
        A ctypes function object representing the C callback.
        """
        ctypes_args = [to_ctypes(ty) for ty in self._sig.args]
        ctypes_restype = to_ctypes(self._sig.return_type)
        functype = ctypes.CFUNCTYPE(ctypes_restype, *ctypes_args)
        return functype(self.address)

    def inspect_llvm(self):
        """
        Return the LLVM IR of the C callback definition.
        """
        return self._library.get_llvm_str()

    @property
    def cache_hits(self):
        return self._cache_hits

    def __repr__(self):
        return "<Numba C callback %r>" % (self.__qualname__,)
