'''
A temporary hack to experiment with a different implementation of Translate
from numba v0.1
'''

from numba.translate import *
import __builtin__
from numba.translate import Translate as _OldTranslate

class Translate(_OldTranslate):
    def __init__(self, func, ret_type='d', arg_types=['d'], module=None, **kws):
        self.func = func
        self.fco = func.func_code
        self.names = self.fco.co_names
        self.varnames = self.fco.co_varnames
        self.constants = self.fco.co_consts
        self.costr = func.func_code.co_code
        # Just the globals we will use
        self._myglobals = {}
        for name in self.names:
            try:
                self._myglobals[name] = func.func_globals[name]
            except KeyError:
                # Assumption here is that any name not in globals or
                # builtins is an attribtue.
                self._myglobals[name] = getattr(__builtin__, name, None)

        ret_type, arg_types = self.map_types(ret_type, arg_types)

        ######## BEGIN CHANGE

        # NOTE: Was seeing weird corner case where
        # llvm.core.Module.new() was not returning a module object,
        # thinking this was caused by compiling the same function
        # twice while the module was being garbage collected, and
        # llvm.core.Module.new() would return whatever was left lying
        # around.  Using the translator address in the module name
        # might fix this.

        # NOTE: It doesn't.  Have to manually flush the llvm-py object cache
        # since not even forcing garbage collection is reliable.

        # NOTE: Fixed?

        # global _ObjectCache
        # setattr(_ObjectCache, '_ObjectCache__instances', WeakValueDictionary())

        if module is not None:
            self.mod = module
        elif not hasattr(type(self), 'mod'):
            type(self).mod = lc.Module.new('default')
            type(self).ee = self._get_ee()

        ######## END CHANGE
        assert isinstance(self.mod, lc.Module), (
            "Expected %r from llvm-py, got instance of type %r, however." %
            (lc.Module, type(self.mod)))
        self._delaylist = [range, xrange, enumerate]
        self.ret_type = ret_type
        self.arg_types = arg_types
        self.setup_func()
        # self.ee = None
        self.ma_obj = None
        self.optimize = kws.pop('optimize', True)
        self.flags = kws

    def setup_func(self):
        # The return type will not be known until the return
        #   function is created.   So, we will need to
        #   walk through the code twice....
        #   Once to get the type of the return, and again to
        #   emit the instructions.
        # For now, we assume the function has been called already
        #   or the return type is otherwise known and passed in
        self.ret_ltype = convert_to_llvmtype(self.ret_type)
        # The arg_ltypes we will be able to get from what is passed in
        argnames = self.fco.co_varnames[:self.fco.co_argcount]
        self.arg_ltypes = [convert_to_llvmtype(x) for x in self.arg_types]
        ty_func = lc.Type.function(self.ret_ltype, self.arg_ltypes)
        ######## BEGIN CHANGE
        orig_func_name = self.func.func_name
        argtyps_decor = '.'.join(str(ty) for ty in self.arg_ltypes)
        self.lfunc = self.mod.add_function(ty_func, '_'.join([orig_func_name, argtyps_decor]))
        ######## END CHANGE
        assert isinstance(self.lfunc, lc.Function), (
            "Expected %r from llvm-py, got instance of type %r, however." %
            (lc.Function, type(self.lfunc)))
        self.nlocals = len(self.fco.co_varnames)
        self._locals = [None] * self.nlocals
        for i, (name, typ) in enumerate(zip(argnames, self.arg_types)):
            assert isinstance(self.lfunc.args[i], lc.Argument), (
                "Expected %r from llvm-py, got instance of type %r, however." %
                (lc.Argument, type(self.lfunc.args[i])))
            self.lfunc.args[i].name = name
            # Store away arguments in locals
            self._locals[i] = Variable(self.lfunc.args[i], typ)
        entry = self.lfunc.append_basic_block('Entry')
        assert isinstance(entry, lc.BasicBlock), (
            "Expected %r from llvm-py, got instance of type %r, however." %
            (lc.BasicBlock, type(entry)))
        self.blocks = {0:entry}
        self.cfg = None
        self.blocks_locals = {}
        self.pending_phis = {}
        self.pending_blocks = {}
        self.stack = []
        self.loop_stack = []

    def _get_ee(self):
        try:
            return self.ee
        except AttributeError:
            return le.EngineBuilder.new(self.mod).opt(3).create()

    def translate(self):
        """Translate the function
        """
        self.cfg = LLVMControlFlowGraph.build_cfg(self.fco, self)
        self.cfg.compute_dataflow()
        if __debug__:
            self.cfg.pprint()
        for i, op, arg in itercode(self.costr):
            name = opcode.opname[op]
            # Change the builder if the line-number
            # is in the list of blocks.
            if i in self.blocks.keys():
                if i > 0:
                    # Emit a branch to link blocks up if the previous
                    # block was not explicitly branched out of...
                    bb_instrs = self.builder.basic_block.instructions
                    if ((len(bb_instrs) == 0) or
                        (not bb_instrs[-1].is_terminator)):
                        self.builder.branch(self.blocks[i])

                    # Copy the locals exiting the soon to be
                    # preceeding basic block.
                    self.blocks_locals[self.crnt_block] = self._locals[:]

                    # Ensure we are playing with locals that might
                    # actually precede the next block.
                    self.check_locals(i)

                self.crnt_block = i
                self.builder = lc.Builder.new(self.blocks[i])
                self.build_phi_nodes(self.crnt_block)
            getattr(self, 'op_'+name)(i, op, arg)

        # Perform code optimization
        if self.optimize:
            ######## BEGIN CHANGE
            pmbldr = lp.PassManagerBuilder.new()
            fpm = lp.FunctionPassManager.new(self.mod)
            pmbldr.opt_level = 3
            pmbldr.populate(fpm)
            ######## END CHANGE

        #if __debug__:
        #    print self.mod
