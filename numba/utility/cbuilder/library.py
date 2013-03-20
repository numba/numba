# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core

def declare(numba_cdef, env, global_module):
    """
    Declare a NumbaCDefinition in the current translation environment.
    """
    # print numba_cdef
    specialized_cdef = numba_cdef(env, global_module)
    lfunc = specialized_cdef.define(global_module) #, optimize=False)
    assert lfunc.module is global_module
    return specialized_cdef, lfunc

registered_utilities = []

def register(utility):
    registered_utilities.append(utility)
    return utility

def load_utilities():
    from . import library
    from . import numbacdef
    from . import refcounting

class CBuilderLibrary(object):
    """
    Library of cbuilder functions.
    """

    def __init__(self):
        self.module = llvm.core.Module.new("cbuilderlib")
        self.funcs = {}

    def declare_registered(self, env):
        "Declare all utilities in our module"
        load_utilities()
        for registered_utility in registered_utilities:
            self.declare(registered_utility, env, self.module)

    def declare(self, numba_cdef, env, llvm_module):
        if numba_cdef not in self.funcs:
            specialized_cdef, lfunc = declare(numba_cdef, env, self.module)
            self.funcs[numba_cdef] = specialized_cdef, lfunc
        else:
            specialized_cdef, lfunc = self.funcs[numba_cdef]

        name = numba_cdef._name_
        lfunc_type = specialized_cdef.signature()
        lfunc = llvm_module.get_or_insert_function(lfunc_type, name)

        return lfunc

    def link(self, llvm_module):
        """
        Link the CBuilder library into the target module.
        """
        llvm_module.link_in(self.module, preserve=True)