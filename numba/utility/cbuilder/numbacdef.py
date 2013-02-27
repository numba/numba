# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import llvm.core
from llvm_cbuilder import builder

from_numba = builder.CStruct.from_numba_struct

class NumbaCDefinition(builder.CDefinition):
    """
    Numba utility simplifying dealing with llvm_cbuilder.
    """

    def __init__(self, env, llvm_module):
        # Environments
        self.env = env
        self.func_env = env.translation.crnt
        self.context = env.context

        self.llvm_module = llvm_module

        self.set_signature(self.env, self.context)

        type(self)._name_ = type(self).__name__
        super(NumbaCDefinition, self).__init__()

    #------------------------------------------------------------------------
    # Convenience Methods
    #------------------------------------------------------------------------

    def external_cfunc(self, func_name):
        "Get a CFunc from an external function"
        signature, lfunc = self.env.context.external_library.declare(
                self.llvm_module,
                func_name)
        assert lfunc.module is self.llvm_module
        return builder.CFunc(self, lfunc)

    def cbuilder_cfunc(self, numba_cdef):
        "Get a CFunc from a NumbaCDefinition"
        lfunc = self.env.context.cbuilder_library.declare(numba_cdef, self.env,
                                                          self.llvm_module)
        assert lfunc.module is self.llvm_module
        return builder.CFunc(self, lfunc)

    #------------------------------------------------------------------------
    # CDefinition stuff
    #------------------------------------------------------------------------

    def signature(self):
        argtypes = [type for name, type in self._argtys_]
        return llvm.core.Type.function(self._retty_, argtypes)

    def __call__(self, module):
        lfunc = super(NumbaCDefinition, self).__call__(module)
        # lfunc.linkage = llvm.core.LINKAGE_LINKONCE_ODR
        return lfunc

    def set_signature(self, env, context):
        """
        Set the cbuilder signature through _argtys_ and optionally the
        _retty_ attributes.
        """
