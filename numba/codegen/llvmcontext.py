import llvm
import llvm.core as lc
import llvm.passes as lp
import llvm.ee as le

from numba import *
from numba import nodes
from numba.typesystem import is_obj, promote_to_native
from numba.codegen.codeutils import llvm_alloca, if_badval
from numba.codegen.debug import *


class LLVMContextManager(object):
    '''TODO: Make this class not a singleton.
             A possible design is to let each Numba Context owns a
             LLVMContextManager.
    '''

    __singleton = None

    def __new__(cls, opt=3, cg=3, inline=1000):
        '''
        opt --- Optimization level for LLVM optimization pass [0 - 3].
        cg  --- Optimization level for code generator [0 - 3].
                Use `3` for SSE support on Intel.
        inline --- Inliner threshold.
        '''
        inst = cls.__singleton
        if not inst:
            inst = object.__new__(cls)
            inst.__initialize(opt, cg, inline)
            cls.__singleton = inst
        return inst

    def __initialize(self, opt, cg, inline):
        assert self.__singleton is None
        m = self.__module = lc.Module.new("numba_executable_module")
        # Create the TargetMachine
        ## FIXME: Use a better CPU feature control
        ##
        #        from llvm.ee import FORCE_DISABLE_AVX
        #        if FORCE_DISABLE_AVX:
        #            features = '-avx'
        #        else:
        #            features = ''
        features = ''
        tm = self.__machine = le.TargetMachine.new(opt=cg, cm=le.CM_JITDEFAULT,
                                                   features=features)
        # Create the ExceutionEngine
        self.__engine = le.EngineBuilder.new(m).create(tm)
        # Build a PassManager which will be used for every module/
        has_loop_vectorizer = llvm.version >= (3, 2)
        passmanagers = lp.build_pass_managers(tm, opt=opt,
                                              inline_threshold=inline,
                                              loop_vectorize=has_loop_vectorizer,
                                              fpm=False)
        self.__pm = passmanagers.pm

    @property
    def module(self):
        return self.__module

    @property
    def execution_engine(self):
        return self.__engine

    @property
    def pass_manager(self):
        return self.__pm

    @property
    def target_machine(self):
        return self.__machine

    def link(self, lfunc):
        if lfunc.module is not self.module:
            # optimize
            self.pass_manager.run(lfunc.module)
            # link module
            func_name = lfunc.name
            #
            #            print 'lfunc.module'.center(80, '-')
            #            print lfunc.module
            #
            #            print 'self.module'.center(80, '-')
            #            print self.module

            # XXX: Better safe than sorry.
            #      Check duplicated function definitions and remove them.
            #      This problem should not exists.
            def is_duplicated_function(f):
                if f.is_declaration:
                    return False
                try:
                    self.module.get_function_named(f.name)
                except llvm.LLVMException as e:
                    return False
                else:
                    return True

            lfunc_module = lfunc.module
            #print "LINKING", lfunc.name, lfunc.module.id
            #print [f.name for f in lfunc.module.functions]
            #print '-----'

            for func in lfunc_module.functions:
                if is_duplicated_function(func):
                    import warnings
                    if func is lfunc:
                        # If the duplicated function is the currently compiling
                        # function, rename it.
                        ct = 0
                        while is_duplicated_function(func):
                            func.name = "%s_duplicated%d" % (func_name, ct)
                            ct += 1
                        warnings.warn("Renamed duplicated function %s to %s" %
                                      (func_name, func.name))
                        func_name = func.name
                    else:
                        # If the duplicated function is not the currently
                        # compiling function, ignore it.
                        # We assume this is a utility function.
                        assert func.linkage == lc.LINKAGE_LINKONCE_ODR, func.name

            self.module.link_in(lfunc_module, preserve=False)
            #
            #            print 'linked'.center(80, '='), lfunc.module, lfunc.name
            #            print self.module

            lfunc = self.module.get_function_named(func_name)

        assert lfunc.module is self.module
        self.verify(lfunc)
        #        print lfunc
        return lfunc

    def get_pointer_to_function(self, lfunc):
        return self.execution_engine.get_pointer_to_function(lfunc)

    def verify(self, lfunc):
        lfunc.module.verify()
        # XXX: remove the following extra checking before release
        for bb in lfunc.basic_blocks:
            for instr in bb.instructions:
                if isinstance(instr, lc.CallOrInvokeInstruction):
                    callee = instr.called_function
                    if callee is not None:
                        assert callee.module is lfunc.module,\
                        "Inter module call for call to %s" % callee.name
