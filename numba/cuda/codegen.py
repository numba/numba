from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc

from numba.core.codegen import BaseCPUCodegen, CodeLibrary
from numba.core import utils
from .cudadrv import nvvm


CUDA_TRIPLE = {32: 'nvptx-nvidia-cuda',
               64: 'nvptx64-nvidia-cuda'}


class CUDACodeLibrary(CodeLibrary):
    # We don't optimize the IR at the function or module level because it is
    # optimized by NVVM after we've passed it on.

    def _optimize_functions(self, ll_module):
        pass

    def _optimize_final_module(self):
        pass

    def _finalize_specific(self):
        # Fix global naming
        for gv in self._final_module.global_variables:
            if '.' in gv.name:
                gv.name = gv.name.replace('.', '_')

    def get_asm_str(self):
        # Return nothing: we can only dump assembly code when it is later
        # generated (in numba.cuda.compiler).
        return None


class JITCUDACodegen(BaseCPUCodegen):
    """
    This codegen implementation for CUDA actually only generates optimized LLVM
    IR.  Generation of PTX code is done separately (see numba.cuda.compiler).
    """

    _library_class = CUDACodeLibrary

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        self._data_layout = nvvm.default_data_layout
        self._target_data = ll.create_target_data(self._data_layout)

    def _create_empty_module(self, name):
        ir_module = lc.Module(name)
        ir_module.triple = CUDA_TRIPLE[utils.MACHINE_BITS]
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        return ir_module

    def _module_pass_manager(self):
        raise NotImplementedError

    def _function_pass_manager(self, llvm_module):
        raise NotImplementedError

    def _add_module(self, module):
        pass
