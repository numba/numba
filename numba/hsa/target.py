from __future__ import print_function, absolute_import
import re
from llvmlite.llvmpy import core as lc
from llvmlite import ir as llvmir
from numba import typing, types, utils, cgutils
from numba.targets.base import BaseContext
from . import codegen
from .hlc import DATALAYOUT

CC_SPIR_KERNEL = "spir_kernel"
CC_SPIR_FUNC = "spir_func"

# -----------------------------------------------------------------------------
# Typing


class HSATypingContext(typing.BaseContext):
    def init(self):
        from . import hsadecl

        self.install(hsadecl.registry)

# -----------------------------------------------------------------------------
# Implementation

VALID_CHARS = re.compile(r'[^a-z0-9]', re.I)


# Address spaces
SPIR_PRIVATE_ADDRSPACE = 0
SPIR_GLOBAL_ADDRSPACE = 1
SPIR_CONSTANT_ADDRSPACE = 2
SPIR_LOCAL_ADDRSPACE = 3
SPIR_GENERIC_ADDRSPACE = 4

SPIR_VERSION = (2, 0)



class HSATargetContext(BaseContext):
    implement_powi_as_math_call = True
    generic_addrspace = SPIR_GENERIC_ADDRSPACE

    def init(self):
        from . import hsaimpl

        self.insert_func_defn(hsaimpl.registry.functions)
        self._internal_codegen = codegen.JITHSACodegen("numba.hsa.jit")
        self._target_data = DATALAYOUT[utils.MACHINE_BITS]

    def jit_codegen(self):
        return self._internal_codegen

    def mangler(self, name, argtypes):
        def repl(m):
            ch = m.group(0)
            return "_%X_" % ord(ch)

        qualified = name + '.' + '.'.join(str(a) for a in argtypes)
        mangled = VALID_CHARS.sub(repl, qualified)
        return mangled

    def prepare_hsa_kernel(self, func, argtypes):
        module = func.module
        func.linkage = 'internal'

        module.data_layout = DATALAYOUT[self.address_size]
        wrapper = self.generate_kernel_wrapper(func, argtypes)
        set_hsa_kernel(wrapper)

        return wrapper

    def mark_hsa_device(self, func):
        # Adapt to SPIR
        # module = func.module
        func.calling_convention = CC_SPIR_FUNC

    def generate_kernel_wrapper(self, func, argtypes):
        module = func.module
        raw_argtys = [self.get_argument_type(ty) for ty in argtypes]
        argtys = []
        for arg in raw_argtys:
            if isinstance(arg, llvmir.PointerType):
                gptr = llvmir.PointerType(arg.pointee,
                                          addrspace=SPIR_GLOBAL_ADDRSPACE)
                argtys.append(gptr)
            else:
                argtys.append(arg)

        fnty = lc.Type.function(lc.Type.void(), argtys)
        wrappername = 'hsaPy_{name}'.format(name=func.name)
        wrapper = module.add_function(fnty, name=wrappername)

        builder = lc.Builder.new(wrapper.append_basic_block(''))

        callargs = []
        for at, av in zip(argtypes, wrapper.args):
            av = self.get_argument_value(builder, at, av)
            callargs.append(av)

        # XXX handle error status
        status, _ = self.call_function(builder, func, types.void, argtypes,
                                       callargs)

        builder.ret_void()
        return wrapper

    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """
        Call the Numba-compiled *callee*, using the same calling
        convention as in get_function_type().
        """
        assert env is None
        retty = callee.args[0].type.pointee
        retval = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(lc.Constant.null(retty), retval)
        args = [self.get_value_as_argument(builder, ty, arg)
                for ty, arg in zip(argtys, args)]
        realargs = [retval] + list(args)
        # Fix addrspace
        fixed = []
        for arg, argty in zip(realargs, callee.function_type.args):
            if isinstance(arg.type, llvmir.PointerType):
                if arg.type.addrspace != argty.addrspace:
                    arg = builder.bitcast(arg, argty)

            fixed.append(arg)

        code = builder.call(callee, fixed)
        status = self.get_return_status(builder, code)
        return status, builder.load(retval)


    def link_dependencies(self, module, depends):
        raise NotImplementedError
        for lib in depends:
            module.link_in(lib, preserve=True)

    def make_constant_array(self, builder, typ, ary):
        """
        Return dummy value.
        """
        #
        # a = self.make_array(typ)(self, builder)
        # return a._getvalue()
        raise NotImplementedError

    def addrspacecast(self, builder, src, addrspace):
        """
        Handle addrspacecast
        """
        ptras = llvmir.PointerType(src.type.pointee, addrspace=addrspace)
        return builder.bitcast(src, ptras)


def set_hsa_kernel(fn):
    """
    Ensure `fn` is usable as a SPIR kernel.
    - Fix calling convention
    - Add metadata
    """
    mod = fn.module

    # Set nounwind
    # fn.add_attribute(lc.ATTR_NO_UNWIND)

    # Set SPIR kernel calling convention
    fn.calling_convention = CC_SPIR_KERNEL

    # Mark kernels
    ocl_kernels = mod.get_or_insert_named_metadata("opencl.kernels")
    ocl_kernels.add(lc.MetaData.get(mod, [fn,
                                          gen_arg_addrspace_md(fn),
                                          gen_arg_access_qual_md(fn),
                                          gen_arg_type(fn),
                                          gen_arg_type_qual(fn),
                                          gen_arg_base_type(fn)]))

    # SPIR version 2.0
    make_constant = lambda x: lc.Constant.int(lc.Type.int(), x)
    spir_version_constant = [make_constant(x) for x in SPIR_VERSION]

    spir_version = mod.get_or_insert_named_metadata("opencl.spir.version")
    if not spir_version.operands:
        spir_version.add(lc.MetaData.get(mod, spir_version_constant))

    ocl_version = mod.get_or_insert_named_metadata("opencl.ocl.version")
    if not ocl_version.operands:
        ocl_version.add(lc.MetaData.get(mod, spir_version_constant))

    ## The following metadata does not seem to be necessary
    # Other metadata
    # empty_md = lc.MetaData.get(mod, ())
    # others = ["opencl.used.extensions",
    #           "opencl.used.optional.core.features",
    #           "opencl.compiler.options"]cat
    #
    # for name in others:
    #     nmd = mod.get_or_insert_named_metadata(name)
    #     if not nmd.operands:
    #         nmd.add(empty_md)


def gen_arg_addrspace_md(fn):
    """
    Generate kernel_arg_addr_space metadata
    """
    mod = fn.module
    fnty = fn.type.pointee
    codes = []

    for a in fnty.args:
        if a.kind == lc.TYPE_POINTER:
            codes.append(SPIR_GLOBAL_ADDRSPACE)
        else:
            codes.append(SPIR_PRIVATE_ADDRSPACE)

    consts = [lc.Constant.int(lc.Type.int(), x) for x in codes]
    name = lc.MetaDataString.get(mod, "kernel_arg_addr_space")
    return lc.MetaData.get(mod, [name] + consts)


def gen_arg_access_qual_md(fn):
    """
    Generate kernel_arg_access_qual metadata
    """
    mod = fn.module
    consts = [lc.MetaDataString.get(mod, "none")] * len(fn.args)
    name = lc.MetaDataString.get(mod, "kernel_arg_access_qual")
    return lc.MetaData.get(mod, [name] + consts)


def gen_arg_type(fn):
    """
    Generate kernel_arg_type metadata
    """
    mod = fn.module
    fnty = fn.type.pointee
    consts = [lc.MetaDataString.get(mod, str(a)) for a in fnty.args]
    name = lc.MetaDataString.get(mod, "kernel_arg_type")
    return lc.MetaData.get(mod, [name] + consts)


def gen_arg_type_qual(fn):
    """
    Generate kernel_arg_type_qual metadata
    """
    mod = fn.module
    fnty = fn.type.pointee
    consts = [lc.MetaDataString.get(mod, "") for _ in fnty.args]
    name = lc.MetaDataString.get(mod, "kernel_arg_type_qual")
    return lc.MetaData.get(mod, [name] + consts)


def gen_arg_base_type(fn):
    """
    Generate kernel_arg_base_type metadata
    """
    mod = fn.module
    fnty = fn.type.pointee
    consts = [lc.MetaDataString.get(mod, str(a)) for a in fnty.args]
    name = lc.MetaDataString.get(mod, "kernel_arg_base_type")
    return lc.MetaData.get(mod, [name] + consts)
