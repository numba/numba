from __future__ import print_function, absolute_import
import re
from llvm.core import (Type, Builder, LINKAGE_INTERNAL, inline_function,
                       Constant, ICMP_EQ)
import llvm.passes as lp
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

    def init(self):
        from . import cudaimpl, libdevice

        self.insert_func_defn(cudaimpl.registry.functions)
        self.insert_func_defn(libdevice.registry.functions)

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
        func.linkage = LINKAGE_INTERNAL
        wrapper = self.generate_kernel_wrapper(func, argtypes)
        func.delete()
        del func

        nvvm.set_cuda_kernel(wrapper)
        nvvm.fix_data_layout(module)

        return wrapper

    def generate_kernel_wrapper(self, func, argtypes):
        module = func.module
        argtys = self.get_arguments(func.type.pointee)
        fnty = Type.function(Type.void(), argtys)
        wrapfn = module.add_function(fnty, name="cudaPy_" + func.name)
        builder = Builder.new(wrapfn.append_basic_block(''))

        # Define error handling variables
        def define_error_gv(postfix):
            gv = module.add_global_variable(Type.int(),
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
            xchg = builder.atomic_cmpxchg(gv_exc, old, status.code,
                                          "monotonic")
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
        inline_function(status.code)

        module.verify()
        return wrapfn

    def link_dependencies(self, module, depends):
        for lib in depends:
            module.link_in(lib, preserve=True)

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
            gl = lmod.add_global_variable(text.type, name=name,
                                          addrspace=nvvm.ADDRSPACE_CONSTANT)
            gl.linkage = LINKAGE_INTERNAL
            gl.global_constant = True
            gl.initializer = text

            constcharptrty = Type.pointer(charty, nvvm.ADDRSPACE_CONSTANT)
            charptr = builder.bitcast(gl, constcharptrty)

        conv = nvvmutils.insert_addrspace_conv(lmod, charty,
                                               nvvm.ADDRSPACE_CONSTANT)
        return builder.call(conv, [charptr])

    def optimize_function(self, func):
        """Run O1 function passes
        """
        fpm = lp.FunctionPassManager.new(func.module)

        lp.PassManagerBuilder.new().populate(fpm)

        fpm.initialize()
        fpm.run(func)
        fpm.finalize()
