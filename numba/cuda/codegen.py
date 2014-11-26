from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc
from numba.targets.codegen import BaseCPUCodegen, CodeLibrary
from .cudadrv import nvvm



class CUDACodeLibrary(CodeLibrary):

    def add_llvm_module(self, ll_module):
        self._final_module.link_in(ll_module)


class JITCUDACodegen(BaseCPUCodegen):

    _library_class = CUDACodeLibrary

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        self._data_layout = nvvm.default_data_layout
        self._target_data = ll.create_target_data(self._data_layout)

    def _create_empty_module(self, name):
        ir_module = lc.Module.new(name)
        # TODO
        # ir_module.triple = ll.get_default_triple()
        return ir_module

    def _module_pass_manager(self):
        raise NotImplementedError

    def _function_pass_manager(self, llvm_module):
        raise NotImplementedError

    def _add_module(self, module):
        #raise NotImplementedError
        self._engine.add_module(module)
