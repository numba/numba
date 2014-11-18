from __future__ import print_function, division, absolute_import

import sys

import llvmlite.llvmpy.core as lc
import llvmlite.llvmpy.ee as le
import llvmlite.llvmpy.passes as lp
import llvmlite.binding as ll
import llvmlite.ir as llvmir

from numba import config


_x86arch = frozenset(['x86', 'i386', 'i486', 'i586', 'i686', 'i786',
                      'i886', 'i986'])

def _is_x86(triple):
    arch = triple.split('-')[0]
    return arch in _x86arch


class BaseCPUCodegen(object):

    _finalized = False

    def __init__(self, module_name):
        self._libraries = []
        self._llvm_module = ll.parse_assembly(
            str(self._create_empty_module(module_name)))
        self._init(self._llvm_module)

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        eb = le.EngineBuilder.new(llvm_module)
        self._customize_engine_builder(eb)
        tm = eb.select_target()
        #tm = le.TargetMachine.new(opt=config.OPT, reloc=le.RELOC_PIC, features='-avx',
                                  #codemodel='jitdefault')
        engine = eb.create(tm)
        engine.add_module(llvm_module)
        data_layout = str(engine.target_data)
        pmb = lp.create_pass_manager_builder(
            opt=config.OPT, loop_vectorize=config.LOOP_VECTORIZE)
        tli = ll.create_target_library_info(llvm_module.triple)

        self._tli = tli
        self._pmb = pmb
        self._tm = tm
        self._engine = engine
        self._data_layout = data_layout

    def _raise_if_finalized(self):
        if self._finalized:
            raise RuntimeError("operation impossible on finalized codegen object")

    def _ensure_finalized(self):
        if not self._finalized:
            self.finalize()

    def _create_empty_module(self, name):
        ir_module = lc.Module.new(name)
        ir_module.triple = ll.get_default_triple()
        return ir_module

    @property
    def target_data(self):
        return self._engine.target_data

    def add_library(self, codegen):
        self._libraries.append(codegen)

    def create_ir_module(self, name):
        self._raise_if_finalized()
        ir_module = self._create_empty_module(name)
        # Enforce data layout to enable layout-specific optimizations
        ir_module.data_layout = self._data_layout
        return ir_module

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()
        assert isinstance(ir_module, llvmir.Module)
        ll_module = ll.parse_assembly(str(ir_module))
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def add_llvm_module(self, ll_module):
        def funcs(mod):
            return [f.name for f in mod.functions]
        fpm = self._function_pass_manager(ll_module)
        # Run function-level optimizations to reduce memory usage and improve
        # module-level optimization.
        for func in ll_module.functions:
            fpm.initialize()
            fpm.run(func)
            fpm.finalize()
        self._llvm_module.link_in(ll_module)

    def finalize(self):
        self._raise_if_finalized()
        self._finalize()
        self._finalized = True

    def get_function(self, name):
        return self._llvm_module.get_function(name)

    def _finalize(self):
        for codegen in self._libraries:
            self._llvm_module.link_in(codegen._llvm_module, preserve=True)
        pm = self._module_pass_manager(self._llvm_module)
        pm.run(self._llvm_module)

        self._finalize_specific()
        self._finalized = True

        if config.DUMP_OPTIMIZED:
            # FIXME
            print(("OPTIMIZED DUMP %s" % fndesc).center(80, '-'))
            print(self._llvm_module)
            print('=' * 80)

        if config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % fndesc).center(80, '-'))
            print(self._tm.emit_assembly(self._llvm_module))
            print('=' * 80)

    def _module_pass_manager(self, llvm_module):
        pm = ll.create_module_pass_manager()
        dl = ll.create_target_data(llvm_module.data_layout)
        dl.add_pass(pm)
        self._tli.add_pass(pm)
        self._tm.add_analysis_passes(pm)
        self._pmb.populate(pm)
        return pm

    def _function_pass_manager(self, llvm_module):
        pm = ll.create_function_pass_manager(llvm_module)
        dl = ll.create_target_data(llvm_module.data_layout)
        dl.add_pass(pm)
        self._tli.add_pass(pm)
        self._tm.add_analysis_passes(pm)
        self._pmb.populate(pm)
        return pm


class AOTCPUCodegen(BaseCPUCodegen):

    def _customize_engine_builder(self, eb):
        pass

    def _finalize_specific(self):
        pass

    def emit_native_object(self):
        self._ensure_finalized()
        return self._tm.emit_object(self._llvm_module)

    def emit_bitcode(self):
        self._ensure_finalized()
        return self._llvm_module.as_bitcode()


class JITCPUCodegen(BaseCPUCodegen):

    def _customize_engine_builder(self, eb):
        if sys.platform.startswith('win32'):
            eb.use_mcjit(False)
        else:
            eb.use_mcjit(True)
        features = []

        # Note: LLVM 3.3 always generates vmovsd (AVX instruction) for
        # mem<->reg move.  The transition between AVX and SSE instruction
        # without proper vzeroupper to reset is causing a serious performance
        # penalty because the SSE register need to save/restore.
        # For now, we will disable the AVX feature for all processor and hope
        # that LLVM 3.5 will fix this issue.
        # features.append('-avx')
        # If this is x86, make sure SSE is supported
        if config.X86_SSE and _is_x86(self._llvm_module.triple):
            features.append('+sse')
            features.append('+sse2')

        # Set feature attributes
        eb.mattrs(','.join(features))

        # Enable JIT debug
        eb.emit_jit_debug(True)

    def _finalize_specific(self):
        self._engine.finalize_object()

    def get_pointer_to_function(self, name):
        self._ensure_finalized()
        func = self.get_function(name)
        return self._engine.get_pointer_to_function(func)

