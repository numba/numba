from __future__ import print_function, absolute_import
import re
import itertools
from llvmlite.llvmpy import core as lc
from numba import typing, types, cgutils, utils
from numba.targets.base import BaseContext
from numba.targets import builtins
from . import codegen
from .hlc import DATALAYOUT

CC_SPIR_KERNEL = "spir_kernel"
CC_SPIR_FUNC = "spir_func"

# -----------------------------------------------------------------------------
# Typing


class HSATypingContext(typing.BaseContext):
    def init(self):
        pass
        # from . import hsadecl
        #
        # self.install(hsadecl.registry)

# -----------------------------------------------------------------------------
# Implementation

VALID_CHARS = re.compile(r'[^a-z0-9]', re.I)


class HSATargetContext(BaseContext):
    implement_powi_as_math_call = True

    def init(self):
        # from . import oclimpl
        #
        # self.insert_func_defn(oclimpl.registry.functions)
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
        argtys = [self.get_argument_type(ty) for ty in argtypes]
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

    # def make_array(self, typ):
    #     return builtins.make_array(typ, addrspace=SPIR_GLOBAL_ADDRSPACE)


class ArgAdaptor(object):
    def __init__(self, ctx, typ):
        self.ctx = ctx
        self.type = typ
        self.adapted_types = tuple(adapt_argument(ctx, self.type))

    def pack(self, builder, args):
        """Pack arguments to match Numba's calling convention
        """
        if isinstance(self.type, types.Array):
            arycls = self.ctx.make_array(self.type)
            ary = arycls(self.ctx, builder)
            ary.data = args[0]
            base = 1
            shape = []
            for i in range(self.type.ndim):
                shape.append(args[base + i])
            base += self.type.ndim
            strides = []
            for i in range(self.type.ndim):
                strides.append(args[base + i])
            base += self.type.ndim
            ary.shape = cgutils.pack_array(builder, shape)
            ary.strides = cgutils.pack_array(builder, strides)
            return ary._getvalue(), base


def adapt_argument(ctx, ty):
    return ty
    # if isinstance(ty, types.Array):
    #     # Handle array
    #     yield lc.Type.pointer(ctx.get_value_type(ty.dtype),
    #                           addr_space=SPIR_GLOBAL_ADDRSPACE)
    #     for i in range(2 * ty.ndim):  # shape + strides
    #         yield ctx.get_value_type(types.intp)
    #
    # elif ty in types.complex_domain:
    #     # Handle complex number
    #     dtype = types.float32 if ty == types.complex64 else types.float64
    #     for _ in range(2):
    #         yield ctx.get_value_type(dtype)
    #
    # else:
    #     yield ctx.get_vaue_type(ty)



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
    print(fn)

    # Mark kernels
    ocl_kernels = mod.get_or_insert_named_metadata("opencl.kernels")
    ocl_kernels.add(lc.MetaData.get(mod, [fn, gen_arg_addrspace_md(fn),
                                          gen_arg_access_qual_md(fn)]))

    # SPIR version
    make_constant = lambda x: lc.Constant.int(lc.Type.int(), x)
    spir_version_constant = [make_constant(x) for x in (1, 2)]

    spir_version = mod.get_or_insert_named_metadata("opencl.spir.version")
    if not spir_version.operands:
        spir_version.add(lc.MetaData.get(mod, spir_version_constant))


    ocl_version = mod.get_or_insert_named_metadata("opencl.ocl.version")
    if not ocl_version.operands:
        ocl_version.add(lc.MetaData.get(mod, spir_version_constant))

    # Other metadata
    empty_md = lc.MetaData.get(mod, ())
    others = ["opencl.used.extensions",
              "opencl.used.optional.core.features",
              "opencl.compiler.options"]

    for name in others:
        nmd = mod.get_or_insert_named_metadata(name)
        if not nmd.operands:
            nmd.add(empty_md)


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
    consts = [lc.MetaDataString.get(mod, "none")]
    name = lc.MetaDataString.get(mod, "kernel_arg_access_qual")
    return lc.MetaData.get(mod, [name] + consts)


# Address spaces
SPIR_PRIVATE_ADDRSPACE = 0
SPIR_GLOBAL_ADDRSPACE = 1
SPIR_CONSTANT_ADDRSPACE = 2
SPIR_LOCAL_ADDRSPACE = 3

