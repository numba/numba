from __future__ import print_function, absolute_import

import sys
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le
from llvm.workaround import avx_support
from numba import _dynfunc, _helperlib, config
from numba.callwrapper import PyCallWrapper
from .base import BaseContext
from numba import utils
from numba.targets import intrinsics, mathimpl, npyimpl
from .options import TargetOptions


def _windows_symbol_hacks_32bits():
    # if we don't have _ftol2, bind _ftol as _ftol2
    ftol2 = le.dylib_address_of_symbol("_ftol2")
    if not ftol2:
        ftol = le.dylib_address_of_symbol("_ftol")
        assert ftol
        le.dylib_add_symbol("_ftol2", ftol)


class CPUContext(BaseContext):
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
        self.is32bit = (tuple.__itemsize__ == 4)

        # map math functions
        self.map_math_functions()

        # Add target specific implementations
        self.insert_func_defn(mathimpl.functions)
        self.insert_func_defn(npyimpl.functions)

    def build_pass_manager(self):
        if config.OPT == 3:
            # This uses the same passes for clang -O3
            pms = lp.build_pass_managers(tm=self.tm, opt=3, loop_vectorize=True,
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
        le.dylib_add_symbol("numba.math.cpow", _helperlib.get_cpow())
        le.dylib_add_symbol("numba.math.sdiv", _helperlib.get_sdiv())
        le.dylib_add_symbol("numba.math.srem", _helperlib.get_srem())
        le.dylib_add_symbol("numba.math.udiv", _helperlib.get_udiv())
        le.dylib_add_symbol("numba.math.urem", _helperlib.get_urem())

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
                imp = getattr(_helperlib, "get_%s" % fname)
                le.dylib_add_symbol(fname, imp())
                self.cmath_provider[fname] = 'indirect'

    def dynamic_map_function(self, func):
        name, ptr = self.native_funcs[func]
        le.dylib_add_symbol(name, ptr)

    def optimize(self, module):
        self.pm.run(module)

    def get_executable(self, func, fndesc):
        """
        Returns
        -------
        (cfunc, fnptr)

        - cfunc
            callable function (Can be None)
        - fnptr
            callable function address

        """
        if self.is32bit:
            dmf = intrinsics.DivmodFixer()
            dmf.run(func.module)

        im = intrinsics.IntrinsicMapping(self)
        im.run(func.module)

        if not fndesc.native:
            self.optimize_pythonapi(func)

        cfunc, fnptr = self.prepare_for_call(func, fndesc)
        return cfunc, fnptr

    def prepare_for_call(self, func, fndesc):
        wrapper, api = PyCallWrapper(self, func.module, func, fndesc).build()
        self.optimize(func.module)

        if config.DEBUG:
            print(func.module)
            print(self.tm.emit_assembly(func.module))

        # Map module.__dict__
        le.dylib_add_symbol(".pymodule.dict." + fndesc.pymod.__name__,
                            id(fndesc.pymod.__dict__))

        # Code gen
        self.engine.add_module(func.module)
        baseptr = self.engine.get_pointer_to_function(func)
        fnptr = self.engine.get_pointer_to_function(wrapper)
        cfunc = _dynfunc.make_function(fndesc.pymod, fndesc.name, fndesc.doc,
                                       fnptr)

        if fndesc.native:
            self.native_funcs[cfunc] = fndesc.mangled_name, baseptr

        return cfunc, fnptr

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


# ----------------------------------------------------------------------------
# TargetOptions

class CPUTargetOptions(TargetOptions):
    OPTIONS = {
        "nopython": bool,
        "forceobj": bool,
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
