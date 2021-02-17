from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc

from numba.core.codegen import BaseCPUCodegen, CodeLibrary
from numba.core import utils
from .cudadrv import nvvm


CUDA_TRIPLE = {32: 'nvptx-nvidia-cuda',
               64: 'nvptx64-nvidia-cuda'}


class CUDACodeLibrary(CodeLibrary):

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        self.modules = []

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

    def add_ir_module(self, mod):
        self.modules.append(mod)

    def add_linking_library(self, library):
        for mod in library.modules:
            if mod not in self.modules:
                self.modules.append(mod)

    def finalize(self):
        # A CUDACodeLibrary isn't a real CodeLibrary that does any code
        # generation, so expecting to do anything with it after finalization is
        # almost certainly an error.
        raise RuntimeError('CUDACodeLibraries cannot be finalized')


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
        nvvm.add_ir_version(ir_module)
        return ir_module

    def _module_pass_manager(self):
        raise NotImplementedError

    def _function_pass_manager(self, llvm_module):
        raise NotImplementedError

    def _add_module(self, module):
        pass
