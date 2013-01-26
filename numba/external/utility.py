import llvm.core

from numba import Py_uintptr_t
from numba import nodes
from numba.external import external

class UtilityFunction(external.ExternalFunction):
    """
    A utility function written in a native language.
    """

    def __init__(self, funcaddr, return_type=None, arg_types=None, **kwargs):
        super(UtilityFunction, self).__init__(return_type, arg_types, **kwargs)
        self.funcaddr = funcaddr

    def llvm_value(self, context):
        lsig = self.signature.to_llvm(context)
        inttype = Py_uintptr_t.to_llvm(context)
        intval = llvm.core.Constant.int(inttype, self.funcaddr)
        return intval.inttoptr(lsig)

    def ast_node(self, context):
        lvalue = self.llvm_value(context)
        nodes.LLVMValueRefNode(self.signature, lvalue)
