from __future__ import print_function, absolute_import
import re
from llvmlite.llvmpy.core import (Type, Builder, LINKAGE_INTERNAL,
                       Constant, ICMP_EQ)
import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll

from numba import typing, types, cgutils, debuginfo, dispatcher
from numba.utils import cached_property
from numba.targets.base import BaseContext
from numba.targets.callconv import MinimalCallConv
from numba.targets import cmathimpl
from numba.typing import cmathdecl

from numba import itanium_mangler
from .cudadrv import nvvm
from . import codegen, nvvmutils
from .decorators import jitdevice


# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.BaseContext):
    def load_additional_registries(self):
        from . import cudadecl, cudamath

        self.install_registry(cudadecl.registry)
        self.install_registry(cudamath.registry)
        self.install_registry(cmathdecl.registry)

    def resolve_value_type(self, val):
        # treat dispatcher object as another device function
        if isinstance(val, dispatcher.Dispatcher):
            try:
                # use cached device function
                val = val.__cudajitdevice
            except AttributeError:
                if not val._can_compile:
                    raise ValueError('using cpu function on device '
                                     'but its compilation is disabled')
                jd = jitdevice(val, debug=val.targetoptions.get('debug'))
                # cache the device function for future use and to avoid
                # duplicated copy of the same function.
                val.__cudajitdevice = jd
                val = jd

        # continue with parent logic
        return super(CUDATypingContext, self).resolve_value_type(val)

# -----------------------------------------------------------------------------
# Implementation

VALID_CHARS = re.compile(r'[^a-z0-9]', re.I)


class CUDATargetContext(BaseContext):
    implement_powi_as_math_call = True
    strict_alignment = True
    DIBuilder = debuginfo.NvvmDIBuilder

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        self._internal_codegen = codegen.JITCUDACodegen("numba.cuda.jit")
        self._target_data = ll.create_target_data(nvvm.default_data_layout)

    def load_additional_registries(self):
        from . import cudaimpl, printimpl, libdevice
        self.install_registry(cudaimpl.registry)
        self.install_registry(printimpl.registry)
        self.install_registry(libdevice.registry)
        self.install_registry(cmathimpl.registry)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        return self._target_data

    @cached_property
    def call_conv(self):
        return CUDACallConv(self)

    def mangler(self, name, argtypes):
        return itanium_mangler.mangle(name, argtypes)

    def prepare_cuda_kernel(self, codelib, fname, argtypes, debug):
        """
        Adapt a code library ``codelib`` with the numba compiled CUDA kernel
        with name ``fname`` and arguments ``argtypes`` for NVVM.
        A new library is created with a wrapper function that can be used as
        the kernel entry point for the given kernel.

        Returns the new code library and the wrapper function.
        """
        library = self.codegen().create_library('')
        library.add_linking_library(codelib)
        wrapper = self.generate_kernel_wrapper(library, fname, argtypes,
                                               debug=debug)
        nvvm.fix_data_layout(library._final_module)
        return library, wrapper

    def generate_kernel_wrapper(self, library, fname, argtypes, debug):
        """
        Generate the kernel wrapper in the given ``library``.
        The function being wrapped have the name ``fname`` and argument types
        ``argtypes``.  The wrapper function is returned.
        """
        arginfo = self.get_arg_packer(argtypes)
        argtys = list(arginfo.argument_types)
        wrapfnty = Type.function(Type.void(), argtys)
        wrapper_module = self.create_module("cuda.kernel.wrapper")
        fnty = Type.function(Type.int(),
                             [self.call_conv.get_return_type(types.pyobject)] + argtys)
        func = wrapper_module.add_function(fnty, name=fname)

        prefixed = itanium_mangler.prepend_namespace(func.name, ns='cudapy')
        wrapfn = wrapper_module.add_function(wrapfnty, name=prefixed)
        builder = Builder(wrapfn.append_basic_block(''))

        # Define error handling variables
        def define_error_gv(postfix):
            gv = wrapper_module.add_global_variable(Type.int(),
                                                    name=wrapfn.name + postfix)
            gv.initializer = Constant.null(gv.type.pointee)
            return gv

        gv_exc = define_error_gv("__errcode__")
        gv_tid = []
        gv_ctaid = []
        for i in 'xyz':
            gv_tid.append(define_error_gv("__tid%s__" % i))
            gv_ctaid.append(define_error_gv("__ctaid%s__" % i))

        callargs = arginfo.from_arguments(builder, wrapfn.args)
        status, _ = self.call_conv.call_function(
            builder, func, types.void, argtypes, callargs)


        if debug:
            # Check error status
            with cgutils.if_likely(builder, status.is_ok):
                builder.ret_void()

            with builder.if_then(builder.not_(status.is_python_exc)):
                # User exception raised
                old = Constant.null(gv_exc.type.pointee)

                # Use atomic cmpxchg to prevent rewriting the error status
                # Only the first error is recorded

                casfnty = lc.Type.function(old.type, [gv_exc.type, old.type,
                                                    old.type])

                casfn = wrapper_module.add_function(casfnty,
                                                    name="___numba_cas_hack")
                xchg = builder.call(casfn, [gv_exc, old, status.code])
                changed = builder.icmp(ICMP_EQ, xchg, old)

                # If the xchange is successful, save the thread ID.
                sreg = nvvmutils.SRegBuilder(builder)
                with builder.if_then(changed):
                    for dim, ptr, in zip("xyz", gv_tid):
                        val = sreg.tid(dim)
                        builder.store(val, ptr)

                    for dim, ptr, in zip("xyz", gv_ctaid):
                        val = sreg.ctaid(dim)
                        builder.store(val, ptr)

        builder.ret_void()

        nvvm.set_cuda_kernel(wrapfn)
        library.add_ir_module(wrapper_module)
        library.finalize()
        wrapfn = library.get_function(wrapfn.name)
        return wrapfn

    def make_constant_array(self, builder, typ, ary):
        """
        Return dummy value.

        XXX: We should be able to move cuda.const.array_like into here.
        """

        a = self.make_array(typ)(self, builder)
        return a._getvalue()

    def insert_const_string(self, mod, string):
        """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """
        text = Constant.stringz(string)
        name = '$'.join(["__conststring__",
                         itanium_mangler.mangle_identifier(string)])
        # Try to reuse existing global
        gv = mod.globals.get(name)
        if gv is None:
            # Not defined yet
            gv = mod.add_global_variable(text.type, name=name,
                                         addrspace=nvvm.ADDRSPACE_CONSTANT)
            gv.linkage = LINKAGE_INTERNAL
            gv.global_constant = True
            gv.initializer = text

        # Cast to a i8* pointer
        charty = gv.type.pointee.element
        return Constant.bitcast(gv,
                                charty.as_pointer(nvvm.ADDRSPACE_CONSTANT))

    def insert_string_const_addrspace(self, builder, string):
        """
        Insert a constant string in the constant addresspace and return a
        generic i8 pointer to the data.

        This function attempts to deduplicate.
        """
        lmod = builder.module
        gv = self.insert_const_string(lmod, string)
        return self.insert_addrspace_conv(builder, gv,
                                          nvvm.ADDRSPACE_CONSTANT)

    def insert_addrspace_conv(self, builder, ptr, addrspace):
        """
        Perform addrspace conversion according to the NVVM spec
        """
        lmod = builder.module
        base_type = ptr.type.pointee
        conv = nvvmutils.insert_addrspace_conv(lmod, base_type, addrspace)
        return builder.call(conv, [ptr])

    def optimize_function(self, func):
        """Run O1 function passes
        """
        pass
        ## XXX skipped for now
        # fpm = lp.FunctionPassManager.new(func.module)
        #
        # lp.PassManagerBuilder.new().populate(fpm)
        #
        # fpm.initialize()
        # fpm.run(func)
        # fpm.finalize()


class CUDACallConv(MinimalCallConv):
    pass
