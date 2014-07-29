from __future__ import print_function, absolute_import

import sys

import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le
from llvm.workaround import avx_support

from numba import _dynfunc, _helperlib, config
from numba.callwrapper import PyCallWrapper
from .base import BaseContext, PYOBJECT
from numba import utils, cgutils, types
from numba.targets import intrinsics, mathimpl, npyimpl, operatorimpl, printimpl
from .options import TargetOptions


def _windows_symbol_hacks_32bits():
    # if we don't have _ftol2, bind _ftol as _ftol2
    ftol2 = le.dylib_address_of_symbol("_ftol2")
    if not ftol2:
        ftol = le.dylib_address_of_symbol("_ftol")
        assert ftol
        le.dylib_add_symbol("_ftol2", ftol)


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

    def init(self):
        self.execmodule = lc.Module.new("numba.exec")
        eb = le.EngineBuilder.new(self.execmodule).opt(3)
        if not avx_support.detect_avx_support():
            eb.mattrs("-avx")
        self.tm = tm = eb.select_target()
        self.engine = eb.create(tm)
        self.pm = self.build_pass_manager()
        self.native_funcs = utils.UniqueDict()
        self.cmath_provider = {}
        self.is32bit = (utils.MACHINE_BITS == 32)

        # map math functions
        self.map_math_functions()
        self.map_numpy_math_functions()

        # Add target specific implementations
        self.insert_func_defn(mathimpl.registry.functions)
        self.insert_func_defn(npyimpl.registry.functions)
        self.insert_func_defn(operatorimpl.registry.functions)
        self.insert_func_defn(printimpl.registry.functions)

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
        argtypes = [self.get_argument_type(aty)
                    for aty in fndesc.argtypes]
        resptr = self.get_return_type(fndesc.restype)
        fnty = lc.Type.function(lc.Type.int(), [resptr, PYOBJECT] + argtypes)
        return fnty

    def declare_function(self, module, fndesc):
        """
        Override parent to handle get_env_argument
        """
        fnty = self.get_function_type(fndesc)
        fn = module.get_or_insert_function(fnty, name=fndesc.mangled_name)
        assert fn.is_declaration
        for ak, av in zip(fndesc.args, self.get_arguments(fn)):
            av.name = "arg.%s" % ak
        self.get_env_argument(fn).name = "env"
        fn.args[0] = ".ret"
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

    def build_pass_manager(self):
        if 0 < config.OPT <= 3:
            # This uses the same passes for clang -O3
            pms = lp.build_pass_managers(tm=self.tm, opt=config.OPT,
                                         loop_vectorize=config.LOOP_VECTORIZE,
                                         fpm=False)
            return pms.pm
        else:
            # This uses minimum amount of passes for fast code.
            # TODO: make it generate vector code
            tm = self.tm
            pm = lp.PassManager.new()
            pm.add(tm.target_data.clone())
            pm.add(lp.TargetLibraryInfo.new(tm.triple))
            # Re-enable for target infomation for vectorization
            # tm.add_analysis_passes(pm)
            passes = '''
            basicaa
            scev-aa
            mem2reg
            sroa
            adce
            dse
            sccp
            instcombine
            simplifycfg
            loops
            indvars
            loop-simplify
            licm
            simplifycfg
            instcombine
            loop-vectorize
            instcombine
            simplifycfg
            globalopt
            globaldce
            '''.split()
            for p in passes:
                pm.add(lp.Pass.new(p))
            return pm

    def map_math_functions(self):
        c_helpers = _helperlib.c_helpers
        for name in ['cpow', 'sdiv', 'srem', 'udiv', 'urem']:
            le.dylib_add_symbol("numba.math.%s" % name, c_helpers[name])
        if sys.platform.startswith('win32') and not le.dylib_address_of_symbol(
                '__ftol2'):
            le.dylib_add_symbol("__ftol2", c_helpers["fptoui"])
        elif sys.platform.startswith(
                'linux') and not le.dylib_address_of_symbol('__fixunsdfdi'):
            le.dylib_add_symbol("__fixunsdfdi", c_helpers["fptoui"])
        # Necessary for Python3
        le.dylib_add_symbol("numba.round", c_helpers["round_even"])
        le.dylib_add_symbol("numba.roundf", c_helpers["roundf_even"])

        # windows symbol hacks
        if sys.platform.startswith('win32') and self.is32bit:
            _windows_symbol_hacks_32bits()

        # List available C-math
        for fname in intrinsics.INTR_MATH:
            if le.dylib_address_of_symbol(fname):
                # Exist
                self.cmath_provider[fname] = 'builtin'
            else:
                # Non-exist
                # Bind from C code
                le.dylib_add_symbol(fname, c_helpers[fname])
                self.cmath_provider[fname] = 'indirect'

    def map_numpy_math_functions(self):
        # add the symbols for numpy math to the execution environment.
        import numba._npymath_exports as npymath

        for sym in npymath.symbols:
            le.dylib_add_symbol(*sym)

    def dynamic_map_function(self, func):
        name, ptr = self.native_funcs[func]
        le.dylib_add_symbol(name, ptr)

    def optimize(self, module):
        self.pm.run(module)

    def finalize(self, func, fndesc):
        """Finalize the compilation.  Called by get_executable().

        - Rewrite intrinsics
        - Fix div & rem instructions on 32bit platform
        - Optimize python API calls
        """
        func.module.target = self.tm.triple

        if self.is32bit:
            dmf = intrinsics.DivmodFixer()
            dmf.run(func.module)

        im = intrinsics.IntrinsicMapping(self)
        im.run(func.module)

        if not fndesc.native:
            self.optimize_pythonapi(func)

    def get_executable(self, func, fndesc, env):
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
        self.finalize(func, fndesc)
        cfunc = self.prepare_for_call(func, fndesc, env)
        return cfunc

    def prepare_for_call(self, func, fndesc, env):
        wrapper, api = PyCallWrapper(self, func.module, func, fndesc,
                                     exceptions=self.exceptions).build()
        self.optimize(func.module)

        if config.DUMP_OPTIMIZED:
            print(("OPTIMIZED DUMP %s" % fndesc).center(80, '-'))
            print(func.module)
            print('=' * 80)

        if config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % fndesc).center(80, '-'))
            print(self.tm.emit_assembly(func.module))
            print('=' * 80)

        # Code gen
        self.engine.add_module(func.module)
        baseptr = self.engine.get_pointer_to_function(func)
        fnptr = self.engine.get_pointer_to_function(wrapper)
        cfunc = _dynfunc.make_function(fndesc.lookup_module(),
                                       fndesc.qualname.split('.')[-1],
                                       fndesc.doc, fnptr, env)

        if fndesc.native:
            self.native_funcs[cfunc] = fndesc.mangled_name, baseptr

        return cfunc

    def optimize_pythonapi(self, func):
        # Simplify the function using
        pms = lp.build_pass_managers(tm=self.tm, opt=1,
                                     mod=func.module)
        fpm = pms.fpm

        fpm.initialize()
        fpm.run(func)
        fpm.finalize()

        # remove extra refct api calls
        remove_refct_calls(func)

    def get_abi_sizeof(self, lty):
        return self.engine.target_data.abi_size(lty)

    def optimize_function(self, func):
        """Run O1 function passes
        """
        pms = lp.build_pass_managers(tm=self.tm, opt=1, pm=False,
                                     mod=func.module)
        fpm = pms.fpm
        fpm.initialize()
        fpm.run(func)
        fpm.finalize()

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
    for inst in bb.instructions:
        if isinstance(inst, lc.CallOrInvokeInstruction):
            fname = inst.called_function.name
            if fname == "Py_IncRef" or fname == "Py_DecRef":
                arg = inst.operands[0]
                if isinstance(arg, lc.ConstantPointerNull):
                    inst.erase_from_parent()


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
