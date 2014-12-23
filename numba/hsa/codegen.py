from __future__ import print_function, absolute_import

from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc
from llvmlite import ir as llvmir
from numba import utils, config
from numba.targets.codegen import BaseCPUCodegen, CodeLibrary
from .hlc import DATALAYOUT, TRIPLE, hlc


class ModuleCollection(object):
    def __init__(self):
        self._modules = []
        self._gvars = {}

    def add(self, ir_module):
        assert isinstance(ir_module, llvmir.Module)
        self._modules.append(ir_module)

    def _load_globals(self, ir_module):
        for gv in ir_module.global_variables:
            glbl = self._gvars.get(gv.name)
            if glbl is None:
                self._gvars[gv.name] = gv
            else:
                if not glbl.is_declaration and not gv.is_declaration:
                    raise NameError(gv.name)
                elif not glbl.is_declaration:
                    self._gvars[gv.name] = glbl
                elif not gv.is_declaration:
                    self._gvars[gv.name] = gv

    @property
    def global_variables(self):
        return self._gvars.values()

    def verify(self):
        return True

    def link_in(self, module):
        assert isinstance(module, ModuleCollection)
        self._modules += module._modules
        for m in module._modules:
            self._load_globals(m)

    def get_function(self, name):
        return self._gvars[name]


class HSACodeLibrary(CodeLibrary):
    def _optimize_functions(self, ll_module):
        pass

    def _optimize_final_module(self):
        pass

    def _finalize_specific(self):
        pass
        # # Fix global naming
        # for gv in self._final_module.global_variables:
        # if '.' in gv.name:
        # gv.name = gv.name.replace('.', '_')
        #
        # def add_ir_module(self, ir_module):
        # """
        # Add a LLVM IR module's contents to this library.
        # """
        #     raise NotImplementedError(ir_module)
        #     self._raise_if_finalized()
        #     assert isinstance(ir_module, llvmir.Module)
        #     self._ir_modules.append(ir_module)
        #
        # def add_llvm_module(self, ll_module):
        #     raise NotImplementedError

        # @property
        # def final_module(self):
        #     assert len(self._ir_modules) == 1
        #     return self._ir_modules[0]
        #
        # def finalize(self):
        #     """
        #     Finalize the library.  After this call, nothing can be added anymore.
        #     Finalization involves various stages of code optimization and
        #     linking.
        #     """
        #     self._raise_if_finalized()
        #     self._optimize_final_module()
        #     self._finalize_specific()
        #     self._finalized = True
        #
        # def get_function(self, name):
        #     raise NotImplementedError


class JITHSACodegen(BaseCPUCodegen):
    _library_class = HSACodeLibrary

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        self._data_layout = DATALAYOUT[utils.MACHINE_BITS]
        self._target_data = ll.create_target_data(self._data_layout)

    def _materialize_module(self, ir_module):
        modcol = ModuleCollection()
        modcol.add(ir_module)
        return modcol

    def _create_empty_module(self, name):
        ir_module = lc.Module.new(name)
        ir_module.triple = TRIPLE
        return ir_module

    def _module_pass_manager(self):
        raise NotImplementedError

    def _function_pass_manager(self, llvm_module):
        raise NotImplementedError

    def _add_module(self, module):
        pass
