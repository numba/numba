from __future__ import print_function, absolute_import

import sys

import llvmlite.llvmpy.core as lc
import llvmlite.binding as ll

from numba import _dynfunc, config
from numba.callwrapper import PyCallWrapper
from .base import BaseContext, PYOBJECT
from numba import utils, cgutils, types
from numba.utils import cached_property
from numba.targets import callconv, codegen, externals, intrinsics, listobj, setobj
from .options import TargetOptions
from numba.runtime import rtsys

# Keep those structures in sync with _dynfunc.c.

class ClosureBody(cgutils.Structure):
    _fields = [('env', types.pyobject)]


class EnvBody(cgutils.Structure):
    _fields = [
        ('globals', types.pyobject),
        ('consts', types.pyobject),
    ]


class CPUContext(BaseContext):
    """
    Changes BaseContext calling convention
    """
    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        self.is32bit = (utils.MACHINE_BITS == 32)
        self._internal_codegen = codegen.JITCPUCodegen("numba.exec")

        # Map external C functions.
        externals.c_math_functions.install(self)
        externals.c_numpy_functions.install(self)

        # Initialize NRT runtime
        rtsys.initialize(self)

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
        # Initialize PRNG state
        randomimpl.random_init()

    @property
    def target_data(self):
        return self._internal_codegen.target_data

    def with_aot_codegen(self, name, **aot_options):
        aot_codegen = codegen.AOTCPUCodegen(name, **aot_options)
        return self.subtarget(_internal_codegen=aot_codegen,
                              aot_mode=True)

    def codegen(self):
        return self._internal_codegen

    @cached_property
    def call_conv(self):
        return callconv.CPUCallConv(self)

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

        cfunc = _dynfunc.make_function(fndesc.lookup_module(),
                                       fndesc.qualname.split('.')[-1],
                                       fndesc.doc, fnptr, env,
                                       # objects to keepalive with the function
                                       (library,)
                                       )

        return cfunc

    def calc_array_sizeof(self, ndim):
        '''
        Calculate the size of an array struct on the CPU target
        '''
        aryty = types.Array(types.int32, ndim, 'A')
        return self.get_abi_sizeof(self.get_value_type(aryty))


# ----------------------------------------------------------------------------
# TargetOptions

class CPUTargetOptions(TargetOptions):
    OPTIONS = {
        "nopython": bool,
        "nogil": bool,
        "forceobj": bool,
        "looplift": bool,
        "boundcheck": bool,
        "_nrt": bool,
        "no_rewrites": bool,
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
