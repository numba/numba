from __future__ import print_function, absolute_import

import re
import numpy as np

from llvmlite.llvmpy import core as lc
from llvmlite import ir as llvmir
from llvmlite import binding as ll

from numba.core import typing, types, utils, cgutils
from numba.core.utils import cached_property
from numba.core import datamodel
from numba.core.base import BaseContext
from numba.core.registry import cpu_target
from numba.core.callconv import MinimalCallConv
from . import codegen


CC_SPIR_KERNEL = "spir_kernel"
CC_SPIR_FUNC = "spir_func"


# -----------------------------------------------------------------------------
# Typing


class DPPLTypingContext(typing.BaseContext):
    def load_additional_registries(self):
        # Declarations for OpenCL API functions and OpenCL Math functions
        from .ocl import ocldecl, mathdecl
        from numba.core.typing import cmathdecl, npydecl

        self.install_registry(ocldecl.registry)
        self.install_registry(mathdecl.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(npydecl.registry)


# -----------------------------------------------------------------------------
# Implementation

VALID_CHARS = re.compile(r'[^a-z0-9]', re.I)


# Address spaces
SPIR_PRIVATE_ADDRSPACE  = 0
SPIR_GLOBAL_ADDRSPACE   = 1
SPIR_CONSTANT_ADDRSPACE = 2
SPIR_LOCAL_ADDRSPACE    = 3
SPIR_GENERIC_ADDRSPACE  = 4

SPIR_VERSION = (2, 0)


LINK_ATOMIC = 111


class GenericPointerModel(datamodel.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        #print("GenericPointerModel:", dmm, fe_type, fe_type.addrspace)
        adrsp = fe_type.addrspace if fe_type.addrspace is not None else SPIR_GENERIC_ADDRSPACE
        #adrsp = SPIR_GENERIC_ADDRSPACE
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer(adrsp)
        super(GenericPointerModel, self).__init__(dmm, fe_type, be_type)


def _init_data_model_manager():
    dmm = datamodel.default_manager.copy()
    dmm.register(types.CPointer, GenericPointerModel)
    return dmm


spirv_data_model_manager = _init_data_model_manager()

def _replace_numpy_ufunc_with_opencl_supported_functions():
    from numba.np.ufunc_db import _ufunc_db as ufunc_db
    from numba.dppl.ocl.mathimpl import lower_ocl_impl, sig_mapper

    ufuncs = [("fabs", np.fabs), ("exp", np.exp), ("log", np.log),
              ("log10", np.log10), ("expm1", np.expm1), ("log1p", np.log1p),
              ("sqrt", np.sqrt), ("sin", np.sin), ("cos", np.cos),
              ("tan", np.tan), ("asin", np.arcsin), ("acos", np.arccos),
              ("atan", np.arctan), ("atan2", np.arctan2), ("sinh", np.sinh),
              ("cosh", np.cosh), ("tanh", np.tanh), ("asinh", np.arcsinh),
              ("acosh", np.arccosh), ("atanh", np.arctanh), ("ldexp", np.ldexp),
              ("floor", np.floor), ("ceil", np.ceil), ("trunc", np.trunc)]

    for name, ufunc in ufuncs:
        for sig in ufunc_db[ufunc].keys():
            if sig in sig_mapper and (name, sig_mapper[sig]) in lower_ocl_impl:
                ufunc_db[ufunc][sig] = lower_ocl_impl[(name, sig_mapper[sig])]


class DPPLTargetContext(BaseContext):
    implement_powi_as_math_call = True
    generic_addrspace = SPIR_GENERIC_ADDRSPACE
    context_name = "dppl.jit"

    def init(self):
        self._internal_codegen = codegen.JITSPIRVCodegen("numba.dppl.jit")
        self._target_data = (ll.create_target_data(codegen
                                .SPIR_DATA_LAYOUT[utils.MACHINE_BITS]))
        # Override data model manager to SPIR model
        self.data_model_manager = spirv_data_model_manager
        self.link_binaries = dict()

        from numba.np.ufunc_db import _lazy_init_db
        import copy
        _lazy_init_db()
        from numba.np.ufunc_db import _ufunc_db as ufunc_db
        self.ufunc_db = copy.deepcopy(ufunc_db)

        from numba.core.cpu import CPUContext
        from numba.core.typing import Context as TypingContext

        self.cpu_context = cpu_target.target_context



    def replace_numpy_ufunc_with_opencl_supported_functions(self):
        from numba.dppl.ocl.mathimpl import lower_ocl_impl, sig_mapper

        ufuncs = [("fabs", np.fabs), ("exp", np.exp), ("log", np.log),
                  ("log10", np.log10), ("expm1", np.expm1), ("log1p", np.log1p),
                  ("sqrt", np.sqrt), ("sin", np.sin), ("cos", np.cos),
                  ("tan", np.tan), ("asin", np.arcsin), ("acos", np.arccos),
                  ("atan", np.arctan), ("atan2", np.arctan2), ("sinh", np.sinh),
                  ("cosh", np.cosh), ("tanh", np.tanh), ("asinh", np.arcsinh),
                  ("acosh", np.arccosh), ("atanh", np.arctanh), ("ldexp", np.ldexp),
                  ("floor", np.floor), ("ceil", np.ceil), ("trunc", np.trunc)]

        for name, ufunc in ufuncs:
            for sig in self.ufunc_db[ufunc].keys():
                if sig in sig_mapper and (name, sig_mapper[sig]) in lower_ocl_impl:
                    self.ufunc_db[ufunc][sig] = lower_ocl_impl[(name, sig_mapper[sig])]


    def load_additional_registries(self):
        from .ocl import oclimpl, mathimpl
        from numba.np import npyimpl
        from . import printimpl

        self.insert_func_defn(oclimpl.registry.functions)
        self.insert_func_defn(mathimpl.registry.functions)
        self.insert_func_defn(npyimpl.registry.functions)
        self.install_registry(printimpl.registry)

        """ To make sure we are calling supported OpenCL math
            functions we will redirect some of NUMBA's NumPy
            ufunc with OpenCL's.
        """
        self.replace_numpy_ufunc_with_opencl_supported_functions()


    @cached_property
    def call_conv(self):
        return DPPLCallConv(self)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        return self._target_data

    def mangler(self, name, argtypes):
        def repl(m):
            ch = m.group(0)
            return "_%X_" % ord(ch)

        qualified = name + '.' + '.'.join(str(a) for a in argtypes)
        mangled = VALID_CHARS.sub(repl, qualified)
        return 'dppl_py_devfn_' + mangled

    def prepare_ocl_kernel(self, func, argtypes):
        module = func.module
        func.linkage = 'linkonce_odr'

        module.data_layout = codegen.SPIR_DATA_LAYOUT[self.address_size]
        wrapper = self.generate_kernel_wrapper(func, argtypes)

        return wrapper

    def mark_ocl_device(self, func):
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
                if lty.addrspace == SPIR_LOCAL_ADDRSPACE:
                    return lty, None
                # DRD : Cast all pointer types to global address space.
                if  lty.addrspace != SPIR_GLOBAL_ADDRSPACE: # jcaraban
                    return (lty.pointee.as_pointer(SPIR_GLOBAL_ADDRSPACE),
                            lty.addrspace)
            return lty, None

        if len(arginfo.argument_types) > 0:
            llargtys, changed = zip(*map(sub_gen_with_global,
                                         arginfo.argument_types))
        else:
            llargtys = changed = ()
        wrapperfnty = lc.Type.function(lc.Type.void(), llargtys)

        wrapper_module = self.create_module("dppl.kernel.wrapper")
        wrappername = 'dpplPy_{name}'.format(name=func.name)

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

        set_dppl_kernel(wrapper)

        #print(str(wrapper_module))
        # Link
        module.link_in(ll.parse_assembly(str(wrapper_module)))
        # To enable inlining which is essential because addrspacecast 1->0 is
        # illegal.  Inlining will optimize the addrspacecast out.
        func.linkage = 'internal'
        wrapper = module.get_function(wrapper.name)
        module.get_function(func.name).linkage = 'internal'
        return wrapper

    def declare_function(self, module, fndesc):
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        fn = module.get_or_insert_function(fnty, name=fndesc.mangled_name)
        fn.attributes.add('alwaysinline')
        ret = super(DPPLTargetContext, self).declare_function(module, fndesc)
        # XXX: Refactor fndesc instead of this special case
        if fndesc.llvm_func_name.startswith('dppl_py_devfn'):
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


    def insert_const_string(self, mod, string):
        """
        This returns a a pointer in the spir generic addrspace.
        """
        text = lc.Constant.stringz(string)

        name = '$'.join(["__conststring__",
                         self.mangler(string, ["str"])])

        # Try to reuse existing global
        try:
            gv = mod.get_global(name)
        except KeyError as e:
            # Not defined yet
            gv = mod.add_global_variable(text.type, name=name,
                                         addrspace=SPIR_GENERIC_ADDRSPACE)
            gv.linkage = 'internal'
            gv.global_constant = True
            gv.initializer = text

        # Cast to a i8* pointer
        charty = gv.type.pointee.element
        return lc.Constant.bitcast(gv,
                               charty.as_pointer(SPIR_GENERIC_ADDRSPACE))


    def addrspacecast(self, builder, src, addrspace):
        """
        Handle addrspacecast
        """
        ptras = llvmir.PointerType(src.type.pointee, addrspace=addrspace)
        return builder.addrspacecast(src, ptras)


def set_dppl_kernel(fn):
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

    spir_version = mod.get_or_insert_named_metadata("dppl.spir.version")
    if not spir_version.operands:
        spir_version.add(lc.MetaData.get(mod, spir_version_constant))

    ocl_version = mod.get_or_insert_named_metadata("dppl.ocl.version")
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


class DPPLCallConv(MinimalCallConv):
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
