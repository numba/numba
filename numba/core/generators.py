"""
Support for lowering generators.
"""

import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor


class GeneratorDescriptor(FunctionDescriptor):
    """
    The descriptor for a generator's next function.
    """
    __slots__ = ()

    @classmethod
    def from_generator_fndesc(cls, func_ir, fndesc, gentype, mangler):
        """
        Build a GeneratorDescriptor for the generator returned by the
        function described by *fndesc*, with type *gentype*.

        The generator inherits the env_name from the *fndesc*.
        All emitted functions for the generator share the same Env.
        """
        assert isinstance(gentype, types.Generator)
        restype = gentype.yield_type
        args = ['gen']
        argtypes = (gentype,)
        qualname = fndesc.qualname + '.next'
        unique_name = fndesc.unique_name + '.next'
        return cls(fndesc.native, fndesc.modname, qualname, unique_name,
                   fndesc.doc, fndesc.typemap, restype, fndesc.calltypes,
                   args, fndesc.kws, argtypes=argtypes, mangler=mangler,
                   inline=False, env_name=fndesc.env_name)

    @property
    def llvm_finalizer_name(self):
        """
        The LLVM name of the generator's finalizer function
        (if <generator type>.has_finalizer is true).
        """
        return 'finalize_' + self.mangled_name


class BaseGeneratorLower:
    """
    Base support class for lowering generators.
    """

    def __init__(self, lower):
        self.context = lower.context
        self.fndesc = lower.fndesc
        self.library = lower.library
        self.func_ir = lower.func_ir
        self.lower = lower

        self.geninfo = lower.generator_info
        self.gentype = self.get_generator_type()
        self.gendesc = GeneratorDescriptor.from_generator_fndesc(
            lower.func_ir, self.fndesc, self.gentype, self.context.mangler)
        self.arg_packer = self.context.get_data_packer(self.fndesc.argtypes)

        self.resume_blocks = {}

    @property
    def call_conv(self):
        return self.lower.call_conv

    def get_pointer(self, builder, genptr, *indices):
        """
        Get a pointer to a specific location in the generator structure.

        Parameters:
        builder : IRBuilder
            The LLVM IR builder.
        genptr : Value
            The pointer to the generator structure.
        indices : tuple
            The indices to access the desired field in the structure.

        Returns:
        Value
            The pointer to the specified field.
        """
        return cgutils.gep_inbounds(builder, genptr, 0, *indices)

    def lower_init_func(self, lower):
        """
        Lower the generator's initialization function (which will fill up
        the passed-by-reference generator structure).
        """
        lower.setup_function(self.fndesc)
        builder = lower.builder

        lower.context.insert_generator(self.gentype, self.gendesc, [self.library])
        lower.extract_function_arguments()
        lower.pre_lower()

        retty = self.context.get_return_type(self.gentype)
        resume_index = self.context.get_constant(types.int32, 0)
        argsty = retty.elements[1]
        statesty = retty.elements[2]

        self.incref_arguments(lower, builder)

        argsval = self.arg_packer.as_data(builder, lower.fnargs)
        statesval = Constant(statesty, None)
        gen_struct = cgutils.make_anonymous_struct(builder, [resume_index, argsval, statesval], retty)
        retval = self.box_generator_struct(lower, gen_struct)

        self.call_conv.return_value(builder, retval)
        lower.post_lower()

    def incref_arguments(self, lower, builder):
        """
        Increment the reference count for all NRT arguments before storing into generator states.
        """
        if self.context.enable_nrt:
            for argty, argval in zip(self.fndesc.argtypes, lower.fnargs):
                self.context.nrt.incref(builder, argty, argval)

    def lower_next_func(self, lower):
        """
        Lower the generator's next() function (which takes the
        passed-by-reference generator structure and returns the next
        yielded value).
        """
        lower.setup_function(self.gendesc)
        builder = lower.builder
        function = lower.function

        genptr, = self.call_conv.get_arguments(function)
        self.arg_packer.load_into(builder, self.get_pointer(builder, genptr, 1), lower.fnargs)

        self.resume_index_ptr = self.get_pointer(builder, genptr, 0, 0)
        self.gen_state_ptr = self.get_pointer(builder, genptr, 0, 2)

        prologue = function.append_basic_block("generator_prologue")

        entry_block_tail = lower.lower_function_body()

        stop_block = self.add_stop_iteration_block(builder, function)

        builder.position_at_end(prologue)
        first_block = self.resume_blocks[0] = lower.blkmap[lower.firstblk]

        switch = builder.switch(builder.load(self.resume_index_ptr), stop_block)
        for index, block in self.resume_blocks.items():
            switch.add_case(index, block)

        builder.position_at_end(entry_block_tail)
        builder.branch(prologue)

    def add_stop_iteration_block(self, builder, function):
        """
        Add a block to handle StopIteration at the entry point of the generator.

        Parameters:
        builder : IRBuilder
            The LLVM IR builder.
        function : Function
            The LLVM function.

        Returns:
        BasicBlock
            The StopIteration block.
        """
        stop_block = function.append_basic_block("stop_iteration")
        builder.position_at_end(stop_block)
        self.call_conv.return_stop_iteration(builder)
        return stop_block

    def lower_finalize_func(self, lower):
        """
        Lower the generator's finalizer.
        """
        fnty = llvmlite.ir.FunctionType(llvmlite.ir.VoidType(),
                                        [self.context.get_value_type(self.gentype)])
        function = cgutils.get_or_insert_function(
            lower.module, fnty, self.gendesc.llvm_finalizer_name)
        entry_block = function.append_basic_block('entry')
        builder = IRBuilder(entry_block)

        genptrty = self.context.get_value_type(self.gentype)
        genptr = builder.bitcast(function.args[0], genptrty)
        self.lower_finalize_func_body(builder, genptr)

    def return_from_generator(self, lower):
        """
        Emit a StopIteration at generator end and mark the generator exhausted.
        """
        indexval = Constant(self.resume_index_ptr.type.pointee, -1)
        lower.builder.store(indexval, self.resume_index_ptr)
        self.call_conv.return_stop_iteration(lower.builder)

    def create_resumption_block(self, lower, index):
        """
        Create a block for resuming the generator at a specific index.

        Parameters:
        lower : Lower
            The lower object containing the function context.
        index : int
            The index at which to resume the generator.
        """
        block_name = f"generator_resume{index}"
        block = lower.function.append_basic_block(block_name)
        lower.builder.position_at_end(block)
        self.resume_blocks[index] = block

    def debug_print(self, builder, msg):
        """
        Print a debug message if DEBUG_JIT is enabled.

        Parameters:
        builder : IRBuilder
            The LLVM IR builder.
        msg : str
            The message to print.
        """
        if config.DEBUG_JIT:
            self.context.debug_print(builder, f"DEBUGJIT: {msg}")


class GeneratorLower(BaseGeneratorLower):
    """
    Support class for lowering nopython generators.
    """

    def get_generator_type(self):
        return self.fndesc.restype

    def box_generator_struct(self, lower, gen_struct):
        return gen_struct

    def lower_finalize_func_body(self, builder, genptr):
        """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """
        self.debug_print(builder, "# generator: finalize")
        if self.context.enable_nrt:
            args_ptr = self.get_pointer(builder, genptr, 1)
            for ty, val in self.arg_packer.load(builder, args_ptr):
                self.context.nrt.decref(builder, ty, val)

        self.debug_print(builder, "# generator: finalize end")
        builder.ret_void()


class PyGeneratorLower(BaseGeneratorLower):
    """
    Support class for lowering object mode generators.
    """

    def get_generator_type(self):
        """
        Compute the actual generator type (the generator function's return
        type is simply "pyobject").
        """
        return types.Generator(
            gen_func=self.func_ir.func_id.func,
            yield_type=types.pyobject,
            arg_types=(types.pyobject,) * self.func_ir.arg_count,
            state_types=(types.pyobject,) * len(self.geninfo.state_vars),
            has_finalizer=True,
        )

    def box_generator_struct(self, lower, gen_struct):
        """
        Box the raw *gen_struct* as a Python object.
        """
        gen_ptr = cgutils.alloca_once_value(lower.builder, gen_struct)
        return lower.pyapi.from_native_generator(gen_ptr, self.gentype, lower.envarg)

    def init_generator_state(self, lower):
        """
        NULL-initialize all generator state variables, to avoid spurious
        decref's on cleanup.
        """
        lower.builder.store(Constant(self.gen_state_ptr.type.pointee, None), self.gen_state_ptr)

    def lower_finalize_func_body(self, builder, genptr):
        """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """
        pyapi = self.context.get_python_api(builder)
        resume_index_ptr = self.get_pointer(builder, genptr, 0, 0)
        resume_index = builder.load(resume_index_ptr)

        need_cleanup = builder.icmp_signed('>', resume_index, Constant(resume_index.type, 0))

        with cgutils.if_unlikely(builder, need_cleanup):
            gen_state_ptr = self.get_pointer(builder, genptr, 2)
            for state_index in range(len(self.gentype.state_types)):
                state_slot = cgutils.gep_inbounds(builder, gen_state_ptr, 0, state_index)
                ty = self.gentype.state_types[state_index]
                val = self.context.unpack_value(builder, ty, state_slot)
                pyapi.decref(val)

        builder.ret_void()


class LowerYield:
    """
    Support class for lowering a particular yield point.
    """

    def __init__(self, lower, yield_point, live_vars):
        self.lower = lower
        self.context = lower.context
        self.builder = lower.builder
        self.genlower = lower.genlower
        self.gentype = self.genlower.gentype

        self.gen_state_ptr = self.genlower.gen_state_ptr
        self.resume_index_ptr = self.genlower.resume_index_ptr
        self.yp = yield_point
        self.inst = self.yp.inst
        self.live_vars = live_vars
        self.live_var_indices = [lower.generator_info.state_vars.index(v) for v in live_vars]

    def lower_yield_suspend(self):
        self.lower.debug_print("# generator suspend")
        self.save_live_vars()
        self.save_resume_index()
        self.lower.debug_print("# generator suspend end")

    def lower_yield_resume(self):
        self.genlower.create_resumption_block(self.lower, self.inst.index)
        self.lower.debug_print("# generator resume")
        self.reload_live_vars()
        self.lower.debug_print("# generator resume end")

    def save_live_vars(self):
        """
        Save live variables in the generator state.
        """
        for state_index, name in zip(self.live_var_indices, self.live_vars):
            state_slot = cgutils.gep_inbounds(self.builder, self.gen_state_ptr, 0, state_index)
            ty = self.gentype.state_types[state_index]
            self.lower._alloca_var(name, self.lower.typeof(name))
            val = self.lower.loadvar(name)
            if self.context.enable_nrt:
                self.context.nrt.incref(self.builder, ty, val)
            self.context.pack_value(self.builder, ty, val, state_slot)

    def save_resume_index(self):
        """
        Save the current resume index in the generator state.
        """
        indexval = Constant(self.resume_index_ptr.type.pointee, self.inst.index)
        self.builder.store(indexval, self.resume_index_ptr)

    def reload_live_vars(self):
        """
        Reload live variables from the generator state.
        """
        for state_index, name in zip(self.live_var_indices, self.live_vars):
            state_slot = cgutils.gep_inbounds(self.builder, self.gen_state_ptr, 0, state_index)
            ty = self.gentype.state_types[state_index]
            val = self.context.unpack_value(self.builder, ty, state_slot)
            self.lower.storevar(val, name)
            if self.context.enable_nrt:
                self.context.nrt.decref(self.builder, ty, val)
