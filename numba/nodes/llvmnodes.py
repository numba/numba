from numba.nodes import *

class LLVMValueRefNode(Node):
    """
    Wrap an LLVM value.
    """

    _fields = []

    def __init__(self, type, llvm_value):
        self.type = type
        self.llvm_value = llvm_value

class BadValue(LLVMValueRefNode):
    def __init__(self, type):
        super(BadValue, self).__init__(type, None)

    def __repr__(self):
        return "bad(%s)" % self.type
