from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc

from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import nvvm


CUDA_TRIPLE = 'nvptx64-nvidia-cuda'


class CUDACodeLibrary(CodeLibrary):

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        self._module = None
        self._linked_modules = []

    def get_llvm_str(self):
        return str(self._module)

    def get_asm_str(self):
        # Return nothing: we can only dump assembly code when it is later
        # generated (in numba.cuda.compiler).
        return None

    def add_ir_module(self, mod):
        self._raise_if_finalized()
        if self._module is not None:
            raise RuntimeError('CUDACodeLibrary only supports one module')
        self._module = mod

    def add_linking_library(self, library):
        library._ensure_finalized()

        # We don't want to allow linking more libraries in after finalization
        # because our linked libraries are modified by the finalization, and we
        # won't be able to finalize again after adding new ones
        self._raise_if_finalized()

        for mod in library.modules:
            if mod not in self._linked_modules:
                self._linked_modules.append(mod)

    def get_function(self, name):
        for fn in self._module.functions:
            if fn.name == name:
                return fn
        raise KeyError(f'Function {name} not found')

    @property
    def modules(self):
        return [self._module] + self._linked_modules

    def finalize(self):
        # Unlike the CPUCodeLibrary, we don't invoke the binding layer here -
        # we only adjust the linkage of functions. Global kernels (with
        # external linkage) have their linkage untouched. Device functions are
        # set linkonce_odr to prevent them appearing in the PTX.

        self._raise_if_finalized()

        # Note in-place modification of the linkage of functions in linked
        # libraries. This presently causes no issues as only device functions
        # are shared across code libraries, so they would always need their
        # linkage set to linkonce_odr. If in a future scenario some code
        # libraries require linkonce_odr linkage of functions in linked
        # modules, and another code library requires another linkage, each code
        # library will need to take its own private copy of its linked modules.
        #
        # See also discussion on PR #890:
        # https://github.com/numba/numba/pull/890
        for mod in self._linked_modules:
            for fn in mod.functions:
                if not fn.is_declaration and fn.linkage != 'external':
                    fn.linkage = 'linkonce_odr'

        self._finalized = True


class JITCUDACodegen(Codegen):
    """
    This codegen implementation for CUDA only generates optimized LLVM IR.
    Generation of PTX code is done separately (see numba.cuda.compiler).
    """

    _library_class = CUDACodeLibrary

    def __init__(self, module_name):
        self._data_layout = nvvm.default_data_layout
        self._target_data = ll.create_target_data(self._data_layout)

    def _create_empty_module(self, name):
        ir_module = lc.Module(name)
        ir_module.triple = CUDA_TRIPLE
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        nvvm.add_ir_version(ir_module)
        return ir_module

    def _add_module(self, module):
        pass
