from __future__ import print_function, absolute_import
import re
import itertools
import llvm.core as lc
from llvmpy.api import llvm
from numba import typing, types
from numba.targets.base import BaseContext

# -----------------------------------------------------------------------------
# Typing


class OCLTypingContext(typing.BaseContext):
    def init(self):
        pass
        # from . import cudadecl, cudamath
        #
        # self.install(cudadecl.registry)
        # self.install(cudamath.registry)

# -----------------------------------------------------------------------------
# Implementation

VALID_CHARS = re.compile(r'[^a-z0-9]', re.I)


class OCLTargetContext(BaseContext):
    implement_powi_as_math_call = True

    def init(self):
        # from . import cudaimpl, libdevice
        #
        # self.insert_func_defn(cudaimpl.registry.functions)
        # self.insert_func_defn(libdevice.registry.functions)
        pass

    def mangler(self, name, argtypes):
        def repl(m):
            ch = m.group(0)
            return "_%X_" % ord(ch)

        qualified = name + '.' + '.'.join(str(a) for a in argtypes)
        mangled = VALID_CHARS.sub(repl, qualified)
        return mangled

    def prepare_ocl_kernel(self, func, argtypes):
        # Adapt to SPIR
        module = func.module
        func.linkage = lc.LINKAGE_INTERNAL
        wrapper = self.generate_kernel_wrapper(func, argtypes)
        func.delete()
        del func

        module.data_layout = DATALAYOUT[self.address_size]
        set_ocl_kernel(wrapper)

        return wrapper

    def generate_kernel_wrapper(self, func, argtypes):
        module = func.module

        adapted = [ArgAdaptor(self, a) for a in argtypes]
        adapted_argtys = [a.adapted_types for a in adapted]
        flattened = tuple(itertools.chain(*adapted_argtys))
        fnty = lc.Type.function(lc.Type.void(), flattened)
        wrapfn = module.add_function(fnty, name="oclPy_" + func.name)
        builder = lc.Builder.new(wrapfn.append_basic_block(''))

        #
        # status, _ = self.call_function(builder, func, types.void, argtypes,
        #                                callargs)
        # TODO handle status

        builder.ret_void()
        del builder
        # force inline
        # lc.inline_function(status.code)

        module.verify()
        return wrapfn

    def link_dependencies(self, module, depends):
        # for lib in depends:
        #     module.link_in(lib, preserve=True)
        raise NotImplementedError

    def make_constant_array(self, builder, typ, ary):
        """
        Return dummy value.

        """
        #
        # a = self.make_array(typ)(self, builder)
        # return a._getvalue()
        raise NotImplementedError


class ArgAdaptor(object):
    def __init__(self, ctx, typ):
        self.ctx = ctx
        self.type = typ
        self.adapted_types = tuple(adapt_argument(ctx, self.type))


def adapt_argument(ctx, ty):
    if isinstance(ty, types.Array):
        # Handle array
        yield lc.Type.pointer(ctx.get_value_type(ty.dtype),
                              addr_space=SPIR_GLOBAL_ADDRSPACE)
        for i in range(2 * ty.ndim):  # shape + strides
            yield ctx.get_value_type(types.intp)

    elif ty in types.complex_domain:
        # Handle complex number
        dtype = types.float32 if ty == types.complex64 else types.float64
        for _ in range(2):
            yield ctx.get_value_type(dtype)

    else:
        yield ctx.get_vaue_type(ty)


DATALAYOUT = {
    32: ("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32"
         ":32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64"
         ":64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512"
         ":512:512-v1024:1024:1024"),
    64: ("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32"
         ":32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64"
         ":64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512"
         ":512:512-v1024:1024:1024")
}


def set_ocl_kernel(fn):
    """
    Ensure `fn` is usable as a SPIR kernel.
    - Fix calling convention
    - Add metadata
    """
    mod = fn.module

    # Set SPIR kernel calling convention
    fn.calling_convention = llvm.CallingConv.ID.SPIR_KERNEL

    # Mark kernels
    ocl_kernels = mod.get_or_insert_named_metadata("opencl.kernels")
    ocl_kernels.add(lc.MetaData.get(mod, [fn, gen_arg_addrspace_md(fn),
                                          gen_arg_access_qual_md(fn)]))

    # SPIR version
    make_constant = lambda x: lc.Constant.int(lc.Type.int(), x)
    spir_version_constant = [make_constant(x) for x in (1, 2)]

    if not mod.get_named_metadata("opencl.spir.version"):
        spir_version = mod.get_or_insert_named_metadata("opencl.spir.version")
        spir_version.add(lc.MetaData.get(mod, spir_version_constant))

    if not mod.get_named_metadata("opencl.ocl.version"):
        ocl_version = mod.get_or_insert_named_metadata("opencl.ocl.version")
        ocl_version.add(lc.MetaData.get(mod, spir_version_constant))

    # Other metadata
    empty_md = lc.MetaData.get(mod, ())
    others = ["opencl.used.extensions",
              "opencl.used.optional.core.features",
              "opencl.compiler.options"]

    for name in others:
        if mod.get_named_metadata(name) is None:
            mod.get_or_insert_named_metadata(name).add(empty_md)


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
