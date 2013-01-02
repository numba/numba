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

class LLVMCBuilderNode(UserNode):
    """
    Instantiate an link in an LLVM cbuilder CDefinition. The CDefinition is
    passed the list of dependence nodes and the list of LLVM value dependencies
    """

    _fields = ["dependencies"]

    def __init__(self, cbuilder_cdefinition, signature, dependencies=None):
        self.cbuilder_cdefinition = cbuilder_cdefinition
        self.type = signature
        self.dependencies = dependencies or []

    def infer_types(self, type_inferer):
        type_inferer.visitchildren(self)
        return self

    def codegen(self, codegen):
        dependencies = codegen.visitlist(self.dependencies)
        cdef = self.cbuilder_cdefinition(self.dependencies, dependencies)
        lfunc = cdef.define(codegen.llvm_module) #, optimize=False)

        from numba import ast_translate
        self.llvm_context = ast_translate.LLVMContextManager()

        # lfunc = self.llvm_context.link(lfunc)
        self.lfunc = lfunc
        codegen.keep_alive(lfunc)

        return lfunc

    @property
    def pointer(self):
        return self.llvm_context.get_pointer_to_function(self.lfunc)
