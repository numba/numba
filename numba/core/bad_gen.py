import llvmlite.binding as ll
import llvmlite.ir as llvmir
import numba.core.cgutils as cgutils
from numba.core.codegen import Codegen, CodeLibrary


class BadCodeLibrary(CodeLibrary):
    """
    A code library that just explodes constantly, for testing purposes
    """

    def __init__(self, codegen: "BadCodegen", name):
        super().__init__(codegen, name)

    def add_linking_library(self, library):
        self._raise_if_finalized()

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()

    def finalize(self):
        self._raise_if_finalized()
        self._finalized = True
        raise NotImplementedError("If you're seeing this, you wrote a test "
                                  "that triggers the compiler during test "
                                  "enumeration. Please do not.")

    def get_defined_functions(self):
        """
        Get all functions defined in the library.  The library must have
        been finalized.
        """
        self._ensure_finalized()

    def get_function(self, name):
        raise KeyError(name)

    def get_llvm_str(self):
        return []

    def get_asm_str(self):
        raise ''


class BadCodegen(Codegen):

    _library_class = BadCodeLibrary

    def __init__(self):
        super().__init__()

    def _add_module(self, module):
        pass

    def _create_empty_module(self, name):
        ir_module = llvmir.Module(cgutils.normalize_ir_text(name))
        ir_module.triple = ll.get_process_triple()
        return ir_module

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
        return ()

    def insert_unresolved_ref(self, builder, fnty, name):
        voidptr = llvmir.IntType(8).as_pointer()
        llvm_mod = builder.module
        try:
            fnptr = llvm_mod.get_function(name)
        except KeyError:
            # Not defined?
            fnptr = llvmir.GlobalVariable(llvm_mod, voidptr, name=name)
            fnptr.linkage = 'external'
        return builder.bitcast(builder.load(fnptr), fnty.as_pointer())
