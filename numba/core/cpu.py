import sys
import platform

import llvmlite.binding as ll
import llvmlite.llvmpy.core as lc
from llvmlite import ir

from numba import _dynfunc
from numba.core.callwrapper import PyCallWrapper
from numba.core.base import BaseContext, PYOBJECT
from numba.core import utils, types, config, cgutils, callconv, codegen, externals, fastmathpass, intrinsics
from numba.core.utils import cached_property
from numba.core.options import TargetOptions
from numba.core.runtime import rtsys
from numba.core.compiler_lock import global_compiler_lock
import numba.core.entrypoints
from numba.core.cpu_options import (ParallelOptions, FastMathOptions,
                                    InlineOptions)
from numba.cpython import setobj, listobj

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
    allow_dynamic_globals = True

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    @global_compiler_lock
    def init(self):
        self.is32bit = (utils.MACHINE_BITS == 32)
        self._internal_codegen = codegen.JITCPUCodegen("numba.exec")

        # Add ARM ABI functions from libgcc_s
        if platform.machine() == 'armv7l':
            ll.load_library_permanently('libgcc_s.so.1')

        # Map external C functions.
        externals.c_math_functions.install(self)

        # Initialize NRT runtime
        rtsys.initialize(self)

        # Initialize additional implementations
        import numba.cpython.unicode
        import numba.cpython.charseq
        import numba.typed.dictimpl
        import numba.experimental.function_type

    def load_additional_registries(self):
        # Add target specific implementations
        from numba.np import npyimpl
        from numba.cpython import cmathimpl, mathimpl, printimpl, randomimpl
        from numba.misc import cffiimpl
        self.install_registry(cmathimpl.registry)
        self.install_registry(cffiimpl.registry)
        self.install_registry(mathimpl.registry)
        self.install_registry(npyimpl.registry)
        self.install_registry(printimpl.registry)
        self.install_registry(randomimpl.registry)

        # load 3rd party extensions
        numba.core.entrypoints.init_all()

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

    def get_env_body(self, builder, envptr):
        """
        From the given *envptr* (a pointer to a _dynfunc.Environment object),
        get a EnvBody allowing structured access to environment fields.
        """
        body_ptr = cgutils.pointer_add(
            builder, envptr, _dynfunc._impl_info['offsetof_env_body'])
        return EnvBody(self, builder, ref=body_ptr, cast_ref=True)

    def get_env_manager(self, builder):
        envgv = self.declare_env_global(builder.module,
                                        self.get_env_name(self.fndesc))
        envarg = builder.load(envgv)
        pyapi = self.get_python_api(builder)
        pyapi.emit_environment_sentry(
            envarg, debug_msg=self.fndesc.env_name,
        )
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

    def build_map(self, builder, dict_type, item_types, items):
        from numba.typed import dictobject

        return dictobject.build_map(self, builder, dict_type, item_types, items)


    def post_lowering(self, mod, library):
        if self.fastmath:
            fastmathpass.rewrite_module(mod, self.fastmath)

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

    def create_cfunc_wrapper(self, library, fndesc, env, call_helper):

        wrapper_module = self.create_module("cfunc_wrapper")
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        wrapper_callee = wrapper_module.add_function(fnty, fndesc.llvm_func_name)

        ll_argtypes = [self.get_value_type(ty) for ty in fndesc.argtypes]
        ll_return_type = self.get_value_type(fndesc.restype)

        wrapty = ir.FunctionType(ll_return_type, ll_argtypes)
        wrapfn = wrapper_module.add_function(wrapty, fndesc.llvm_cfunc_wrapper_name)
        builder = ir.IRBuilder(wrapfn.append_basic_block('entry'))

        status, out = self.call_conv.call_function(
            builder, wrapper_callee, fndesc.restype, fndesc.argtypes,
            wrapfn.args, attrs=('noinline',))

        with builder.if_then(status.is_error, likely=False):
            # If (and only if) an error occurred, acquire the GIL
            # and use the interpreter to write out the exception.
            pyapi = self.get_python_api(builder)
            gil_state = pyapi.gil_ensure()
            self.call_conv.raise_error(builder, pyapi, status)
            cstr = self.insert_const_string(builder.module, repr(self))
            strobj = pyapi.string_from_string(cstr)
            pyapi.err_write_unraisable(strobj)
            pyapi.decref(strobj)
            pyapi.gil_release(gil_state)

        builder.ret(out)
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
        library.codegen.set_env(self.get_env_name(fndesc), env)
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
        "boundscheck": lambda X: bool(X) if X is not None else None,
        "debug": bool,
        "_nrt": bool,
        "no_rewrites": bool,
        "no_cpython_wrapper": bool,
        "no_cfunc_wrapper": bool,
        "fastmath": FastMathOptions,
        "error_model": str,
        "parallel": ParallelOptions,
        "inline": InlineOptions,
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
