from __future__ import print_function, absolute_import

import re

from llvmlite.llvmpy import core as lc
from llvmlite import ir as llvmir
from llvmlite import binding as ll

from numba import typing, types, utils, cgutils
from numba.utils import cached_property
from numba import datamodel
from numba.targets.base import BaseContext
from numba.targets.callconv import MinimalCallConv
from . import codegen
from .hlc import DATALAYOUT

CC_SPIR_KERNEL = "spir_kernel"
CC_SPIR_FUNC = "spir_func"


# -----------------------------------------------------------------------------
# Typing


class HSATypingContext(typing.BaseContext):
    def load_additional_registries(self):
        from . import hsadecl, mathdecl

        self.install_registry(hsadecl.registry)
        self.install_registry(mathdecl.registry)


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


class GenericPointerModel(datamodel.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        adrsp = SPIR_GENERIC_ADDRSPACE
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer(adrsp)
        super(GenericPointerModel, self).__init__(dmm, fe_type, be_type)


def _init_data_model_manager():
    dmm = datamodel.default_manager.copy()
    dmm.register(types.CPointer, GenericPointerModel)
    return dmm


hsa_data_model_manager = _init_data_model_manager()


class HSATargetContext(BaseContext):
    implement_powi_as_math_call = True
    generic_addrspace = SPIR_GENERIC_ADDRSPACE

    def init(self):
        self._internal_codegen = codegen.JITHSACodegen("numba.hsa.jit")
        self._target_data = DATALAYOUT[utils.MACHINE_BITS]
        # Override data model manager
        self.data_model_manager = hsa_data_model_manager

    def load_additional_registries(self):
        from . import hsaimpl, mathimpl

        self.insert_func_defn(hsaimpl.registry.functions)
        self.insert_func_defn(mathimpl.registry.functions)

    @cached_property
    def call_conv(self):
        return HSACallConv(self)

    def codegen(self):
        return self._internal_codegen

    def mangler(self, name, argtypes):
        def repl(m):
            ch = m.group(0)
            return "_%X_" % ord(ch)

        qualified = name + '.' + '.'.join(str(a) for a in argtypes)
        mangled = VALID_CHARS.sub(repl, qualified)
        return 'hsapy_devfn_' + mangled

    def prepare_hsa_kernel(self, func, argtypes):
        module = func.module
        func.linkage = 'linkonce_odr'

        module.data_layout = DATALAYOUT[self.address_size]
        wrapper = self.generate_kernel_wrapper(func, argtypes)

        return wrapper

    def mark_hsa_device(self, func):
        # Adapt to SPIR
        # module = func.module
        func.calling_convention = CC_SPIR_FUNC
        func.linkage = 'linkonce_odr'
        return func

    def generate_kernel_wrapper(self, func, argtypes):
        module = func.module
        arginfo = self.get_arg_packer(argtypes)

        def sub_gen_with_global(lty):
            if isinstance(lty, llvmir.PointerType):
                return (lty.pointee.as_pointer(SPIR_GLOBAL_ADDRSPACE),
                        lty.addrspace)
            return lty, None

        if len(arginfo.argument_types) > 0:
            llargtys, changed = zip(*map(sub_gen_with_global,
                                         arginfo.argument_types))
        else:
            llargtys = changed = ()
        wrapperfnty = lc.Type.function(lc.Type.void(), llargtys)

        wrapper_module = self.create_module("hsa.kernel.wrapper")
        wrappername = 'hsaPy_{name}'.format(name=func.name)

        argtys = list(arginfo.argument_types)
        fnty = lc.Type.function(lc.Type.int(),
                                [self.call_conv.get_return_type(
                                    types.pyobject)] + argtys)

        func = wrapper_module.add_function(fnty, name=func.name)
        func.calling_convention = CC_SPIR_FUNC

        wrapper = wrapper_module.add_function(wrapperfnty, name=wrappername)

        builder = lc.Builder(wrapper.append_basic_block(''))

        # Adjust address space of each kernel argument
        fixed_args = []
        for av, adrsp in zip(wrapper.args, changed):
            if adrsp is not None:
                casted = self.addrspacecast(builder, av, adrsp)
                fixed_args.append(casted)
            else:
                fixed_args.append(av)

        callargs = arginfo.from_arguments(builder, fixed_args)

        # XXX handle error status
        status, _ = self.call_conv.call_function(builder, func, types.void,
                                                 argtypes, callargs)
        builder.ret_void()

        set_hsa_kernel(wrapper)

        # Link
        module.link_in(ll.parse_assembly(str(wrapper_module)))
        # To enable inlining which is essential because addrspacecast 1->0 is
        # illegal.  Inlining will optimize the addrspacecast out.
        func.linkage = 'internal'
        wrapper = module.get_function(wrapper.name)
        module.get_function(func.name).linkage = 'internal'
        return wrapper

    def declare_function(self, module, fndesc):
        ret = super(HSATargetContext, self).declare_function(module, fndesc)
        # XXX: Refactor fndesc instead of this special case
        if fndesc.llvm_func_name.startswith('hsapy_devfn'):
            ret.calling_convention = CC_SPIR_FUNC
        return ret

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
        return builder.addrspacecast(src, ptras)


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
        if cgutils.is_pointer(a):
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


class HSACallConv(MinimalCallConv):
    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """
        Call the Numba-compiled *callee*.
        """
        assert env is None
        retty = callee.args[0].type.pointee
        retvaltmp = cgutils.alloca_once(builder, retty)
        # initialize return value
        builder.store(cgutils.get_null_value(retty), retvaltmp)

        arginfo = self.context.get_arg_packer(argtys)
        args = arginfo.as_arguments(builder, args)
        realargs = [retvaltmp] + list(args)
        code = builder.call(callee, realargs)
        status = self._get_return_status(builder, code)
        retval = builder.load(retvaltmp)
        out = self.context.get_returned_value(builder, resty, retval)
        return status, out
