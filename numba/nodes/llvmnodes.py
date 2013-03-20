# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.nodes import *
from numba import functions

class LLVMValueRefNode(ExprNode):
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

    def __init__(self, env, cbuilder_cdefinition, signature, dependencies=None):
        self.env = env
        self.llvm_context = env.llvm_context
        self.cbuilder_cdefinition = cbuilder_cdefinition
        self.type = signature
        self.dependencies = dependencies or []

    def infer_types(self, type_inferer):
        type_inferer.visitchildren(self)
        return self

    def codegen(self, codegen):
        func_env = self.env.translation.crnt

        dependencies = codegen.visitlist(self.dependencies)
        cdef = self.cbuilder_cdefinition(self.dependencies, dependencies)

        self.lfunc = cdef.define(func_env.llvm_module) #, optimize=False)
        functions.keep_alive(func_env.func, self.lfunc)

        return self.lfunc

    @property
    def pointer(self):
        return self.llvm_context.get_pointer_to_function(self.lfunc)
