import os

from numba.utility.cbuilder import numbacdef

def get_module(env):
    func_env = env.translation.crnt
    if func_env:
        mod = func_env.llvm_module
    else:
        # FunctionEnvironment is None for the numba wrapper function
        mod = env.llvm_context.module

    return mod

def declare(numba_cdef, env):
    """
    Declare a NumbaCDefinition in the current translation environment.
    """
    # print numba_cdef
    global_module = env.llvm_context.module
    specialized_cdef = numba_cdef(env, global_module)
    lfunc = specialized_cdef(global_module)
    return specialized_cdef, lfunc

registered_utilities = []

def register(utility):
    registered_utilities.append(utility)
    return utility

class CBuilderLibrary(object):
    """
    Library of cbuilder functions.
    """

    def __init__(self):
        self.funcs = {}

    def declare_registered(self, env):
        for registered_utility in registered_utilities:
            self.declare(registered_utility, env)

    def declare(self, numba_cdef, env):
        if numba_cdef not in self.funcs:
            specialized_cdef, lfunc = declare(numba_cdef, env)
            self.funcs[numba_cdef] = specialized_cdef, lfunc
        else:
            specialized_cdef, lfunc = self.funcs[numba_cdef]

        global_module = env.llvm_context.module
        llvm_module = get_module(env)

        if global_module is not llvm_module:
            # Generate external declaration for module
            name = numba_cdef._name_
            lfunc_type = specialized_cdef.signature()
            lfunc = llvm_module.get_or_insert_function(lfunc_type, name)
        else:
            pass
            # os.write(1, "Using global module\n")

        return lfunc
