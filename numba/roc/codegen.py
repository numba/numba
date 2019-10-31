from __future__ import print_function, absolute_import

from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc
from numba import utils
from numba.targets.codegen import BaseCPUCodegen, CodeLibrary
from .hlc import DATALAYOUT, TRIPLE, hlc


class HSACodeLibrary(CodeLibrary):
    def _optimize_functions(self, ll_module):
        pass

    def _optimize_final_module(self):
        pass

    def _finalize_specific(self):
        pass

    def get_asm_str(self):
        """
        Get the human-readable assembly.
        """
        m = hlc.Module()
        m.load_llvm(str(self._final_module))
        out = m.finalize()
        return str(out.hsail)


class JITHSACodegen(BaseCPUCodegen):
    _library_class = HSACodeLibrary

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        self._data_layout = DATALAYOUT[utils.MACHINE_BITS]
        self._target_data = ll.create_target_data(self._data_layout)

    def _create_empty_module(self, name):
        ir_module = lc.Module(name)
        ir_module.triple = TRIPLE
        return ir_module

    def _module_pass_manager(self):
        raise NotImplementedError

    def _function_pass_manager(self, llvm_module):
        raise NotImplementedError

    def _add_module(self, module):
        pass
