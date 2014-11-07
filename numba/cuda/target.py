from __future__ import print_function, absolute_import
import re
from llvmlite.llvmpy.core import (Type, Builder, LINKAGE_INTERNAL,
                       Constant, ICMP_EQ)
import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll

from numba import typing, types, cgutils
from numba.targets.base import BaseContext
from .cudadrv import nvvm
from . import nvvmutils


# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.BaseContext):
    def init(self):
        from . import cudadecl, cudamath

        self.install(cudadecl.registry)
        self.install(cudamath.registry)

# -----------------------------------------------------------------------------
# Implementation

VALID_CHARS = re.compile(r'[^a-z0-9]', re.I)


class CUDATargetContext(BaseContext):
    implement_powi_as_math_call = True
    strict_alignment = True

    def init(self):
        from . import cudaimpl, libdevice

        self.insert_func_defn(cudaimpl.registry.functions)
        self.insert_func_defn(libdevice.registry.functions)
        self.target_data = ll.create_target_data(nvvm.default_data_layout)

    def mangler(self, name, argtypes):
        def repl(m):
            ch = m.group(0)
            return "_%X_" % ord(ch)

        qualified = name + '.' + '.'.join(str(a) for a in argtypes)
        mangled = VALID_CHARS.sub(repl, qualified)
        return mangled

    def prepare_cuda_kernel(self, func, argtypes):
        # Adapt to CUDA LLVM
        module = func.module
        wrapper = self.generate_kernel_wrapper(func, argtypes)
        func.linkage = LINKAGE_INTERNAL
        nvvm.fix_data_layout(module)
        return wrapper

    def generate_kernel_wrapper(self, func, argtypes):
        module = func.module
        argtys = [self.get_argument_type(ty) for ty in argtypes]
        wrapfnty = Type.function(Type.void(), argtys)
        wrapper_module = self.create_module("cuda.kernel.wrapper")
        fnty = Type.function(Type.int(),
                             [self.get_return_type(types.pyobject)] + argtys)
        func = wrapper_module.add_function(fnty, name=func.name)
        wrapfn = wrapper_module.add_function(wrapfnty, name="cudaPy_" + func.name)
        builder = Builder.new(wrapfn.append_basic_block(''))

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

        callargs = []
        for at, av in zip(argtypes, wrapfn.args):
            av = self.get_argument_value(builder, at, av)
            callargs.append(av)

        status, _ = self.call_function(builder, func, types.void, argtypes,
                                       callargs)

        # Check error status
        with cgutils.if_likely(builder, status.ok):
            builder.ret_void()

        with cgutils.ifthen(builder, builder.not_(status.exc)):
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
            with cgutils.ifthen(builder, changed):
                for dim, ptr, in zip("xyz", gv_tid):
                    val = sreg.tid(dim)
                    builder.store(val, ptr)

                for dim, ptr, in zip("xyz", gv_ctaid):
                    val = sreg.ctaid(dim)
                    builder.store(val, ptr)

        builder.ret_void()
        # force inline
        # inline_function(status.code)
        nvvm.set_cuda_kernel(wrapfn)
        module.link_in(ll.parse_assembly(str(wrapper_module)))
        module.verify()

        wrapfn = module.get_function(wrapfn.name)
        return wrapfn

    def make_constant_array(self, builder, typ, ary):
        """
        Return dummy value.

        XXX: We should be able to move cuda.const.array_like into here.
        """

        a = self.make_array(typ)(self, builder)
        return a._getvalue()

    def insert_string_const_addrspace(self, builder, string):
        """
        Insert a constant string in the constant addresspace and return a
        generic i8 pointer to the data.

        This function attempts to deduplicate.
        """
        lmod = builder.basic_block.function.module
        text = Constant.stringz(string)
        name = "__conststring__.%s" % string
        charty = Type.int(8)

        for gv in lmod.global_variables:
            if gv.name == name and gv.type.pointee == text.type:
                break
        else:
            gv = lmod.add_global_variable(text.type, name=name,
                                          addrspace=nvvm.ADDRSPACE_CONSTANT)
            gv.linkage = LINKAGE_INTERNAL
            gv.global_constant = True
            gv.initializer = text

        constcharptrty = Type.pointer(charty, nvvm.ADDRSPACE_CONSTANT)
        charptr = builder.bitcast(gv, constcharptrty)

        conv = nvvmutils.insert_addrspace_conv(lmod, charty,
                                               nvvm.ADDRSPACE_CONSTANT)
        return builder.call(conv, [charptr])

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

    def get_abi_sizeof(self, lty):
        return self.target_data.abi_size(lty)
