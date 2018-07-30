from __future__ import print_function, absolute_import

import sys

import llvmlite.llvmpy.core as lc
import llvmlite as ll

from numba import _dynfunc, config
from numba.callwrapper import PyCallWrapper
from .base import BaseContext, PYOBJECT
from numba import utils, cgutils, types, itanium_mangler
from numba.utils import cached_property
from numba.targets import callconv, codegen, externals, intrinsics, listobj, setobj, registry
from .options import TargetOptions
from numba.runtime import rtsys
from . import fastmathpass
from llvmlite.llvmpy.core import (Type, Builder, LINKAGE_INTERNAL, Constant, ICMP_EQ)
from llvmlite import ir
from ctypes import *


# Keep those structures in sync with _dynfunc.c.

class ClosureBody(cgutils.Structure):
    _fields = [('env', types.pyobject)]


class EnvBody(cgutils.Structure):
    _fields = [
        ('globals', types.pyobject),
        ('consts', types.pyobject),
    ]


class CSAContext(BaseContext):
    """
    Changes BaseContext calling convention
    """
    allow_dynamic_globals = True

    # Overrides
    def create_module(self, name):
        if config.DEBUG_CSA:
            print("CSAContext::create_module")
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        if config.DEBUG_CSA:
            print("CSAContext::init")
        self.is32bit = (utils.MACHINE_BITS == 32)
        self._internal_codegen = codegen.JITCSACodegen("numba.csa.exec")

        # Map external C functions.
        #externals.c_math_functions.install(self)

        # Initialize NRT runtime
        #rtsys.initialize(self)

    def load_additional_registries(self):
        # Add target specific implementations
        from . import (cffiimpl, cmathimpl, mathimpl, npyimpl, operatorimpl,
                       printimpl, randomimpl)
        self.install_registry(cmathimpl.registry)
        self.install_registry(cffiimpl.registry)
        self.install_registry(mathimpl.registry)
        self.install_registry(npyimpl.registry)
        self.install_registry(operatorimpl.registry)
        self.install_registry(printimpl.registry)
        self.install_registry(randomimpl.registry)
        self.install_registry(randomimpl.registry)

    @property
    def target_data(self):
        if config.DEBUG_CSA:
            print("CSAContext::target_data")
        return self._internal_codegen.target_data

    def with_aot_codegen(self, name, **aot_options):
        if config.DEBUG_CSA:
            print("CSAContext::with_aot_codegen")
        aot_codegen = codegen.AOTCSACodegen(name, **aot_options)
        return self.subtarget(_internal_codegen=aot_codegen,
                              aot_mode=True)

    def codegen(self):
        if config.DEBUG_CSA:
            print("CSAContext::codegen")
        return self._internal_codegen

    @cached_property
    def call_conv(self):
        return callconv.CSACallConv(self)

    def get_env_from_closure(self, builder, clo):
        """
        From the pointer *clo* to a _dynfunc.Closure, get a pointer
        to the enclosed _dynfunc.Environment.
        """
        with cgutils.if_unlikely(builder, cgutils.is_null(builder, clo)):
            self.debug_print(builder, "Fatal error: missing _dynfunc.Closure")
            builder.unreachable()

        clo_body_ptr = cgutils.pointer_add(
            builder, clo, _dynfunc._impl_info['offsetof_closure_body'])
        clo_body = ClosureBody(self, builder, ref=clo_body_ptr, cast_ref=True)
        return clo_body.env

    def get_env_body(self, builder, envptr):
        """
        From the given *envptr* (a pointer to a _dynfunc.Environment object),
        get a EnvBody allowing structured access to environment fields.
        """
        body_ptr = cgutils.pointer_add(
            builder, envptr, _dynfunc._impl_info['offsetof_env_body'])
        return EnvBody(self, builder, ref=body_ptr, cast_ref=True)

    def get_env_manager(self, builder, envarg=None):
        envarg = envarg or self.call_conv.get_env_argument(builder.function)
        pyapi = self.get_python_api(builder)
        pyapi.emit_environment_sentry(envarg)
        env_body = self.get_env_body(builder, envarg)
        return pyapi.get_env_manager(self.environment, env_body, envarg)

    def get_generator_state(self, builder, genptr, return_type):
        """
        From the given *genptr* (a pointer to a _dynfunc.Generator object),
        get a pointer to its state area.
        """
        return cgutils.pointer_add(
            builder, genptr, _dynfunc._impl_info['offsetof_generator_state'],
            return_type=return_type)

    def build_list(self, builder, list_type, items):
        """
        Build a list from the Numba *list_type* and its initial *items*.
        """
        return listobj.build_list(self, builder, list_type, items)

    def build_set(self, builder, set_type, items):
        """
        Build a set from the Numba *set_type* and its initial *items*.
        """
        return setobj.build_set(self, builder, set_type, items)

    def post_lowering(self, mod, library):
        if self.enable_fastmath:
            fastmathpass.rewrite_module(mod)

        if self.is32bit:
            # 32-bit machine needs to replace all 64-bit div/rem to avoid
            # calls to compiler-rt
            intrinsics.fix_divmod(mod)

        library.add_linking_library(rtsys.library)

    def create_cpython_wrapper(self, library, fndesc, env, call_helper,
                               release_gil=False):
        wrapper_module = self.create_module("wrapper")
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        wrapper_callee = wrapper_module.add_function(fnty, fndesc.llvm_func_name)
        builder = PyCallWrapper(self, wrapper_module, wrapper_callee,
                                fndesc, env, call_helper=call_helper,
                                release_gil=release_gil)
        builder.build()
        library.add_ir_module(wrapper_module)

    def get_executable(self, library, fndesc, env):
        """
        Returns
        -------
        (cfunc, fnptr)

        - cfunc
            callable function (Can be None)
        - fnptr
            callable function address
        - env
            an execution environment (from _dynfunc)
        """
        # Code generation
        baseptr = library.get_pointer_to_function(fndesc.llvm_func_name)
        fnptr = library.get_pointer_to_function(fndesc.llvm_cpython_wrapper_name)

        # Note: we avoid reusing the original docstring to avoid encoding
        # issues on Python 2, see issue #1908
        doc = "compiled wrapper for %r" % (fndesc.qualname,)
        cfunc = _dynfunc.make_function(fndesc.lookup_module(),
                                       fndesc.qualname.split('.')[-1],
                                       doc, fnptr, env,
                                       # objects to keepalive with the function
                                       (library,)
                                       )

        return cfunc

    def calc_array_sizeof(self, ndim):
        '''
        Calculate the size of an array struct on the CSA target
        '''
        aryty = types.Array(types.int32, ndim, 'A')
        return self.get_abi_sizeof(self.get_value_type(aryty))

    def prepare_csa_kernel(self, codelib, fname, argtypes):
        #pdb.set_trace()
        """
        Adapt a code library ``codelib`` with the numba compiled CSA kernel
        with name ``fname`` and arguments ``argtypes`` for CSA.
        A new library is created with a wrapper function that can be used as
        the kernel entry point for the given kernel.

        Returns the new code library and the wrapper function.
        """
        csa_asm_name = fname.split("$")[0] + '.kernel.csa_asm.s'
        if config.DEBUG_CSA:
            print("prepare_csa_kernel", codelib, type(codelib))
            print("fname", fname, type(fname))
            print("argtypes", argtypes, type(argtypes))
            print("csa_asm_name", csa_asm_name, type(csa_asm_name))

        codelib.get_asm_str(csa_asm_name)
        library = self.codegen().create_library('')
        #library.add_linking_library(codelib)
        wrapper, wrapfnty, wrapper_library = self.generate_kernel_wrapper(library, fname, argtypes, csa_asm_name)
        return library, wrapper, wrapfnty, wrapper_library

    def generate_kernel_wrapper(self, library, fname, argtypes, csa_asm_name):
        """
        Generate the kernel wrapper in the given ``library``.
        The function being wrapped have the name ``fname`` and argument types
        ``argtypes``.  The wrapper function is returned.
        """
        library.finalize()
        cput = registry.dispatcher_registry['cpu'].targetdescr 
        context = cput.target_context

        arginfo = context.get_arg_packer(argtypes)
        argtys = list(arginfo.argument_types)
        #wrapfnty = Type.function(Type.void(), argtys)
        wrapfnty = context.call_conv.get_function_type(types.pyobject, argtypes)
#        wrapfnty = Type.function(Type.void(), [context.call_conv.get_return_type(types.pyobject)] + argtys)
        wrapper_module = context.create_module("csa.kernel.wrapper")
#        fnty = Type.function(Type.int(),
#                             [self.call_conv.get_return_type(types.pyobject)] + argtys)
#        func = wrapper_module.add_function(fnty, name=fname)

        prefixed = itanium_mangler.prepend_namespace(fname, ns='csapy')
        wrapfn = wrapper_module.add_function(wrapfnty, name=prefixed)
        builder = Builder(wrapfn.append_basic_block(''))

        if config.DEBUG_CSA:
            print("wrapfn", wrapfn)
            print("wrapfn.args", wrapfn.args)
    #        callargs = arginfo.from_arguments(builder, wrapfn.args)
            print("arginfo", arginfo)
            print("argtys", argtys)
            print("wrapfnty", wrapfnty)
            print("wrapper_module", wrapper_module)
    #        print("fnty", fnty)
            print("fname", fname)
            print("prefixed", prefixed)
            print("wrapfn", wrapfn)
            print("builder", builder)

        ll.binding.load_library_permanently("/home/taanders/numba/numba_csa2/numba/numba/targets/libgeneric.so")

        csa_asm_bytes = cgutils.make_bytearray((csa_asm_name + '\00').encode('ascii'))
        csa_asm_global = cgutils.global_constant(builder.module, "csa_asm_name", csa_asm_bytes)

        fname_bytes = cgutils.make_bytearray((fname + '\00').encode('ascii'))
        fname_global = cgutils.global_constant(builder.module, "fname", fname_bytes)

        intp_t = context.get_value_type(types.intp)
        intp_ptr_t = lc.Type.pointer(intp_t)
        byte_t = lc.Type.int(8)
        byte_ptr_t = lc.Type.pointer(byte_t) 
        llint_t = lc.Type.int(64)
        llint_ptr_t = lc.Type.pointer(llint_t)

        num_args = len(wrapfn.args)
        argarray = cgutils.alloca_once(builder, intp_t, size=context.get_constant(types.intp, num_args), name="argarray")
        for i in range(num_args):
            if config.DEBUG_CSA:
                print(wrapfn.args[i].name, type(wrapfn.args[i].name))
                print(wrapfn.args[i].type, type(wrapfn.args[i].type))
            if isinstance(wrapfn.args[i].type, ir.PointerType):
                if config.DEBUG_CSA:
                    print("ptrtoint", wrapfn.args[i].type)
                    cgutils.printf(builder, "csa_wrapper ptrtoint %p\n", wrapfn.args[i])
                builder.store(builder.ptrtoint(wrapfn.args[i], intp_t), builder.gep(argarray, [context.get_constant(types.intp, i)]))
            else:
                if config.DEBUG_CSA:
                    print("bitcast", wrapfn.args[i].type)
                    cgutils.printf(builder, "csa_wrapper bitcast %d\n", wrapfn.args[i])
                builder.store(builder.bitcast(wrapfn.args[i], intp_t), builder.gep(argarray, [context.get_constant(types.intp, i)]))

        pci_fnty = ir.FunctionType(intp_t, [byte_ptr_t, byte_ptr_t, intp_t, llint_ptr_t])
        pci = builder.module.get_or_insert_function(pci_fnty, name="python_csa_invoke")
        builder.call(pci,
                     [
                         builder.bitcast(csa_asm_global, byte_ptr_t),
                         builder.bitcast(fname_global, byte_ptr_t),
                         context.get_constant(types.intp, num_args),
                         argarray
                     ]
                    )

        builder.ret(callconv.RETCODE_OK)

        if config.DEBUG_CSA:
            print("final wrapper_module", wrapper_module)
        wrapper_library = context.codegen().create_library('')
        wrapper_library.add_ir_module(wrapper_module)
        wrapper_library.finalize()
        wrapfn = wrapper_library.get_function(wrapfn.name)
        return wrapfn, wrapfnty, wrapper_library

class ParallelOptions(object):
    """
    Options for controlling auto parallelization.
    """
    def __init__(self, value):
        if isinstance(value, bool):
            self.enabled = value
            self.comprehension = value
            self.reduction = value
            self.setitem = value
            self.numpy = value
            self.stencil = value
            self.fusion = value
            self.prange = value
        elif isinstance(value, dict):
            self.enabled = True
            self.comprehension = value.pop('comprehension', True)
            self.reduction = value.pop('reduction', True)
            self.setitem = value.pop('setitem', True)
            self.numpy = value.pop('numpy', True)
            self.stencil = value.pop('stencil', True)
            self.fusion = value.pop('fusion', True)
            self.prange = value.pop('prange', True)
            if value:
                raise NameError("Unrecognized parallel options: %s" % value.keys())
        else:
            raise ValueError("Expect parallel option to be either a bool or a dict")


# ----------------------------------------------------------------------------
# TargetOptions

class CSATargetOptions(TargetOptions):
    OPTIONS = {
        "nopython": bool,
        "nogil": bool,
        "forceobj": bool,
        "looplift": bool,
        "boundcheck": bool,
        "debug": bool,
        "_nrt": bool,
        "no_rewrites": bool,
        "no_cpython_wrapper": bool,
        "fastmath": bool,
        "error_model": str,
        "parallel": ParallelOptions,
    }


# ----------------------------------------------------------------------------
# Internal

def remove_refct_calls(func):
    """
    Remove redundant incref/decref within on a per block basis
    """
    for bb in func.basic_blocks:
        remove_null_refct_call(bb)
        remove_refct_pairs(bb)


def remove_null_refct_call(bb):
    """
    Remove refct api calls to NULL pointer
    """
    pass
    ## Skipped for now
    # for inst in bb.instructions:
    #     if isinstance(inst, lc.CallOrInvokeInstruction):
    #         fname = inst.called_function.name
    #         if fname == "Py_IncRef" or fname == "Py_DecRef":
    #             arg = inst.args[0]
    #             print(type(arg))
    #             if isinstance(arg, lc.ConstantPointerNull):
    #                 inst.erase_from_parent()


def remove_refct_pairs(bb):
    """
    Remove incref decref pairs on the same variable
    """

    didsomething = True

    while didsomething:
        didsomething = False

        increfs = {}
        decrefs = {}

        # Mark
        for inst in bb.instructions:
            if isinstance(inst, lc.CallOrInvokeInstruction):
                fname = inst.called_function.name
                if fname == "Py_IncRef":
                    arg = inst.operands[0]
                    increfs[arg] = inst
                elif fname == "Py_DecRef":
                    arg = inst.operands[0]
                    decrefs[arg] = inst

        # Sweep
        for val in increfs.keys():
            if val in decrefs:
                increfs[val].erase_from_parent()
                decrefs[val].erase_from_parent()
                didsomething = True
