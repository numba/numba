from __future__ import print_function
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le
from numba import _dynfunc
from numba.callwrapper import PyCallWrapper
from .base import BaseContext


class CPUContext(BaseContext):
    def init(self):
        self.execmodule = lc.Module.new("numba.exec")
        eb = le.EngineBuilder.new(self.execmodule).opt(3)
        self.tm = tm = eb.select_target()
        self.engine = eb.create(tm)

        pms = lp.build_pass_managers(tm=self.tm, loop_vectorize=True, opt=2,
                                     fpm=False)
        self.pm = pms.pm

        # self.pm = lp.PassManager.new()
        # self.pm.add(lp.Pass.new("mem2reg"))
        # self.pm.add(lp.Pass.new("simplifycfg"))

    def optimize(self, module):
        self.pm.run(module)

    def get_executable(self, func, fndesc):
        wrapper = PyCallWrapper(self, func.module, func, fndesc).build()
        self.optimize(func.module)
        print(func.module)
        self.engine.add_module(func.module)
        fnptr = self.engine.get_pointer_to_function(wrapper)

        func = _dynfunc.make_function(fndesc.pymod, fndesc.name, fndesc.doc,
                                      fnptr)
        return func

