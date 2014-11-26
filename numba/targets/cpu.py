from __future__ import print_function, absolute_import

import sys

import llvmlite.llvmpy.core as lc
import llvmlite.llvmpy.ee as le
import llvmlite.binding as ll

from numba import _dynfunc, _helperlib, config
from numba.callwrapper import PyCallWrapper
from .base import BaseContext, PYOBJECT
from numba import utils, cgutils, types
from numba.targets import (
    codegen, intrinsics, cmathimpl, mathimpl, npyimpl, operatorimpl, printimpl)
from .options import TargetOptions


def _add_missing_symbol(symbol, addr):
    """Add missing symbol into LLVM internal symtab
    """
    if not ll.address_of_symbol(symbol):
        ll.add_symbol(symbol, addr)

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
    _data_layout = None

    # Overrides
    def create_module(self, name):
        mod = lc.Module.new(name)
        mod.triple = ll.get_default_triple()

        if self._data_layout:
            mod.data_layout = self._data_layout
        return mod

    def init(self):
        self._internal_codegen = codegen.JITCPUCodegen("numba.exec")
        #self._internal_library = self._internal_codegen.create_library("common")

        self.native_funcs = utils.UniqueDict()
        self.cmath_provider = {}
        self.is32bit = (utils.MACHINE_BITS == 32)

        # map math functions
        self.map_math_functions()
        self.map_numpy_math_functions()

        # Add target specific implementations
        self.insert_func_defn(cmathimpl.registry.functions)
        self.insert_func_defn(mathimpl.registry.functions)
        self.insert_func_defn(npyimpl.registry.functions)
        self.insert_func_defn(operatorimpl.registry.functions)
        self.insert_func_defn(printimpl.registry.functions)

    @property
    def target_data(self):
        return self._internal_codegen.target_data

    def aot_codegen(self, name):
        return codegen.AOTCPUCodegen(name)

    def jit_codegen(self):
        return self._internal_codegen

    def get_function_type(self, fndesc):
        """
        Get the implemented Function type for the high-level *fndesc*.
        Some parameters can be added or shuffled around.
        This is kept in sync with call_function() and get_arguments().

        Calling Convention
        ------------------
        (Same return value convention as BaseContext target.)
        Returns: -2 for return none in native function;
                 -1 for failure with python exception set;
                  0 for success;
                 >0 for user error code.
        Return value is passed by reference as the first argument.

        The 2nd argument is a _dynfunc.Environment object.
        It MUST NOT be used if the function is in nopython mode.

        Actual arguments starts at the 3rd argument position.
        Caller is responsible to allocate space for return value.
        """
        return self.get_function_type2(fndesc.restype, fndesc.argtypes)

    def get_function_type2(self, restype, argtypes):
        """
        Get the implemented Function type for the high-level *fndesc*.
        Some parameters can be added or shuffled around.
        This is kept in sync with call_function() and get_arguments().

        Calling Convention
        ------------------
        (Same return value convention as BaseContext target.)
        Returns: -2 for return none in native function;
                 -1 for failure with python exception set;
                  0 for success;
                 >0 for user error code.
        Return value is passed by reference as the first argument.

        The 2nd argument is a _dynfunc.Environment object.
        It MUST NOT be used if the function is in nopython mode.

        Actual arguments starts at the 3rd argument position.
        Caller is responsible to allocate space for return value.
        """
        argtypes = [self.get_argument_type(aty)
                    for aty in argtypes]
        resptr = self.get_return_type(restype)
        fnty = lc.Type.function(lc.Type.int(), [resptr, PYOBJECT] + argtypes)
        return fnty

    def declare_function(self, module, fndesc):
        """
        Override parent to handle get_env_argument
        """
        fnty = self.get_function_type(fndesc)
        fn = module.get_or_insert_function(fnty, name=fndesc.llvm_func_name)
        assert fn.is_declaration
        for ak, av in zip(fndesc.args, self.get_arguments(fn)):
            av.name = "arg.%s" % ak
        self.get_env_argument(fn).name = "env"
        fn.args[0].name = "ret"
        return fn

    def get_arguments(self, func):
        """Override parent to handle enviroment argument
        Get the Python-level arguments of LLVM *func*.
        See get_function_type() for the calling convention.
        """
        return func.args[2:]

    def get_env_argument(self, func):
        """
        Get the environment argument of LLVM *func* (which can be
        a declaration).
        """
        return func.args[1]

    def call_function(self, builder, callee, resty, argtys, args, env=None):
        """
        Call the Numba-compiled *callee*, using the same calling
        convention as in get_function_type().
        """
        if env is None:
            # This only works with functions that don't use the environment
            # (nopython functions).
            env = lc.Constant.null(PYOBJECT)
        retty = callee.args[0].type.pointee
        retval = cgutils.alloca_once(builder, retty)
        # initialize return value to zeros
        builder.store(lc.Constant.null(retty), retval)

        args = [self.get_value_as_argument(builder, ty, arg)
                for ty, arg in zip(argtys, args)]
        realargs = [retval, env] + list(args)
        code = builder.call(callee, realargs)
        status = self.get_return_status(builder, code)
        return status, builder.load(retval)

    def get_env_from_closure(self, builder, clo):
        """
        From the pointer *clo* to a _dynfunc.Closure, get a pointer
        to the enclosed _dynfunc.Environment.
        """
        clo_body_ptr = cgutils.pointer_add(
            builder, clo, _dynfunc._impl_info['offset_closure_body'])
        clo_body = ClosureBody(self, builder, ref=clo_body_ptr, cast_ref=True)
        return clo_body.env

    def get_env_body(self, builder, envptr):
        """
        From the given *envptr* (a pointer to a _dynfunc.Environment object),
        get a EnvBody allowing structured access to environment fields.
        """
        body_ptr = cgutils.pointer_add(
            builder, envptr, _dynfunc._impl_info['offset_env_body'])
        return EnvBody(self, builder, ref=body_ptr, cast_ref=True)

    def map_math_functions(self):
        c_helpers = _helperlib.c_helpers
        for name in ['cpow', 'sdiv', 'srem', 'udiv', 'urem']:
            ll.add_symbol("numba.math.%s" % name, c_helpers[name])

        if sys.platform.startswith('win32') and self.is32bit:
            # For Windows XP __ftol2 is not defined, we will just use
            # __ftol as a replacement.
            # On Windows 7, this is not necessary but will work anyway.
            ftol = ll.address_of_symbol('_ftol')
            _add_missing_symbol("_ftol2", ftol)

        elif sys.platform.startswith('linux') and self.is32bit:
            _add_missing_symbol("__fixunsdfdi", c_helpers["fptoui"])
            _add_missing_symbol("__fixunssfdi", c_helpers["fptouif"])

        # Necessary for Python3
        ll.add_symbol("numba.round", c_helpers["round_even"])
        ll.add_symbol("numba.roundf", c_helpers["roundf_even"])

        # List available C-math
        for fname in intrinsics.INTR_MATH:
            if ll.address_of_symbol(fname):
                # Exist
                self.cmath_provider[fname] = 'builtin'
            else:
                # Non-exist
                # Bind from C code
                ll.add_symbol(fname, c_helpers[fname])
                self.cmath_provider[fname] = 'indirect'

    def map_numpy_math_functions(self):
        # add the symbols for numpy math to the execution environment.
        import numba._npymath_exports as npymath

        for sym in npymath.symbols:
            ll.add_symbol(*sym)

    def dynamic_map_function(self, func):
        name, ptr = self.native_funcs[func]
        ll.add_symbol(name, ptr)

    def remove_native_function(self, func):
        """
        Remove internal references to nonpython mode function *func*.
        KeyError is raised if the function isn't known to us.
        """
        name, ptr = self.native_funcs.pop(func)
        # If the symbol wasn't redefined, NULL it out.
        # (otherwise, it means the corresponding Python function was
        #  re-compiled, and the new target is still alive)
        if ll.address_of_symbol(name) == ptr:
            ll.add_symbol(name, 0)

    def post_lowering(self, func):
        mod = func.module

        if (sys.platform.startswith('linux') or
                sys.platform.startswith('win32')):
            intrinsics.fix_powi_calls(mod)

        if self.is32bit:
            # 32-bit machine needs to replace all 64-bit div/rem to avoid
            # calls to compiler-rt
            intrinsics.fix_divmod(mod)

    def create_cpython_wrapper(self, library, fndesc, exceptions):
        #func = codegen.get_function(fndesc.llvm_func_name)
        wrapper_module = self.create_module("wrapper")
        fnty = self.get_function_type(fndesc)
        wrapper_callee = wrapper_module.add_function(fnty, fndesc.llvm_func_name)
        PyCallWrapper(self, wrapper_module, wrapper_callee,
                      fndesc, exceptions=exceptions).build()
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
        func = library.get_function(fndesc.llvm_func_name)
        wrapper = library.get_function(fndesc.llvm_cpython_wrapper_name)

        # FIXME
        #if config.DUMP_OPTIMIZED:
            #print(("OPTIMIZED DUMP %s" % fndesc).center(80, '-'))
            #print(module)
            #print('=' * 80)

        #if config.DUMP_ASSEMBLY:
            #print(("ASSEMBLY %s" % fndesc).center(80, '-'))
            #print(self.tm.emit_assembly(module))
            #print('=' * 80)

        # Code generation
        baseptr = library.get_pointer_to_function(func.name)
        fnptr = library.get_pointer_to_function(wrapper.name)

        cfunc = _dynfunc.make_function(fndesc.lookup_module(),
                                       fndesc.qualname.split('.')[-1],
                                       fndesc.doc, fnptr, env)

        if fndesc.native:
            self.native_funcs[cfunc] = fndesc.llvm_func_name, baseptr

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
        "forceobj": bool,
        "looplift": bool,
        "wraparound": bool,
        "boundcheck": bool,
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
