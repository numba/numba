"""
Support for lowering generators.
"""
from __future__ import print_function, division, absolute_import

from llvmlite.llvmpy.core import Constant, Type, Builder

from . import cgutils, types
from .funcdesc import FunctionDescriptor


class GeneratorDescriptor(FunctionDescriptor):
    """
    The descriptor for a generator's next function.
    """
    __slots__ = ()

    @classmethod
    def from_generator_fndesc(cls, interp, fndesc, gentype, mangler):
        """
        Build a GeneratorDescriptor for the generator returned by the
        function described by *fndesc*, with type *gentype*.
        """
        assert isinstance(gentype, types.Generator)
        restype = gentype.yield_type
        args = ['gen']
        argtypes = [gentype]
        qualname = fndesc.qualname + '.next'
        unique_name = fndesc.unique_name + '.next'
        self = cls(fndesc.native, fndesc.modname, qualname, unique_name,
                   fndesc.doc, fndesc.typemap, restype, fndesc.calltypes,
                   args, fndesc.kws, argtypes=argtypes, mangler=mangler,
                   inline=True)
        return self

    @property
    def llvm_finalizer_name(self):
        """
        The LLVM name of the generator's finalizer function
        (if <generator type>.has_finalizer is true).
        """
        return 'finalize_' + self.mangled_name


class BaseGeneratorLower(object):
    """
    Base support class for lowering generators.
    """

    def __init__(self, lower):
        self.context = lower.context
        self.fndesc = lower.fndesc
        self.library = lower.library
        self.call_conv = lower.call_conv
        self.interp = lower.interp

        self.geninfo = lower.generator_info
        self.gentype = self.get_generator_type()
        self.gendesc = GeneratorDescriptor.from_generator_fndesc(
            lower.interp, self.fndesc, self.gentype, self.context.mangler)

        self.resume_blocks = {}

    def lower_init_func(self, lower):
        """
        Lower the generator's initialization function (which will fill up
        the passed-by-reference generator structure).
        """
        lower.setup_function(self.fndesc)
        builder = lower.builder

        # Insert the generator into the target context in order to allow
        # calling from other Numba-compiled functions.
        lower.context.insert_generator(self.gentype, self.gendesc,
                                       [self.library])

        # Init argument values
        lower.extract_function_arguments()

        lower.pre_lower()

        # Initialize the return structure (i.e. the generator structure).
        retty = self.context.get_return_type(self.gentype)
        # Structure index #0: the initial resume index (0 == start of generator)
        resume_index = self.context.get_constant(types.int32, 0)
        # Structure index #1: the function arguments
        argsty = retty.elements[1]

        # Incref all NRT objects before storing into generator states
        if self.context.enable_nrt:
            for argty, argval in zip(self.fndesc.argtypes, lower.fnargs):
                self.context.nrt_incref(builder, argty, argval)

        argsval = cgutils.make_anonymous_struct(builder, lower.fnargs,
                                                argsty)
        gen_struct = cgutils.make_anonymous_struct(builder,
                                                   [resume_index, argsval],
                                                   retty)
        retval = self.box_generator_struct(lower, gen_struct)
        self.call_conv.return_value(builder, retval)

        lower.post_lower()

    def lower_next_func(self, lower):
        """
        Lower the generator's next() function (which takes the
        passed-by-reference generator structure and returns the next
        yielded value).
        """
        lower.setup_function(self.gendesc)
        assert self.gendesc.argtypes[0] == self.gentype
        builder = lower.builder
        function = lower.function

        # Extract argument values and other information from generator struct
        genptr, = self.call_conv.get_arguments(function)
        for i, ty in enumerate(self.gentype.arg_types):
            argptr = cgutils.gep(builder, genptr, 0, 1, i)
            lower.fnargs[i] = self.context.unpack_value(builder, ty, argptr)
        self.resume_index_ptr = cgutils.gep(builder, genptr, 0, 0,
                                            name='gen.resume_index')
        self.gen_state_ptr = cgutils.gep(builder, genptr, 0, 2,
                                         name='gen.state')

        prologue = function.append_basic_block("generator_prologue")

        # Lower the generator's Python code
        entry_block_tail = lower.lower_function_body()

        # Add block for StopIteration on entry
        stop_block = function.append_basic_block("stop_iteration")
        builder.position_at_end(stop_block)
        self.call_conv.return_stop_iteration(builder)

        # Add prologue switch to resume blocks
        builder.position_at_end(prologue)
        # First Python block is also the resume point on first next() call
        first_block = self.resume_blocks[0] = lower.blkmap[lower.firstblk]

        # Create front switch to resume points
        switch = builder.switch(builder.load(self.resume_index_ptr),
                                stop_block)
        for index, block in self.resume_blocks.items():
            switch.add_case(index, block)

        # Close tail of entry block
        builder.position_at_end(entry_block_tail)
        builder.branch(prologue)

        # Run target specific post lowering transformation
        self.context.post_lowering(function)

    def lower_finalize_func(self, lower):
        """
        Lower the generator's finalizer.
        """
        fnty = Type.function(Type.void(),
                             [self.context.get_value_type(self.gentype)])
        function = lower.module.get_or_insert_function(
            fnty, name=self.gendesc.llvm_finalizer_name)
        entry_block = function.append_basic_block('entry')
        builder = Builder.new(entry_block)

        genptrty = self.context.get_value_type(self.gentype)
        genptr = builder.bitcast(function.args[0], genptrty)
        self.lower_finalize_func_body(builder, genptr)

    def return_from_generator(self, lower):
        """
        Emit a StopIteration at generator end and mark the generator exhausted.
        """
        indexval = Constant.int(self.resume_index_ptr.type.pointee, -1)
        lower.builder.store(indexval, self.resume_index_ptr)
        self.call_conv.return_stop_iteration(lower.builder)

    def create_resumption_block(self, lower, index):
        block_name = "generator_resume%d" % (index,)
        block = lower.function.append_basic_block(block_name)
        lower.builder.position_at_end(block)
        self.resume_blocks[index] = block


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
        if self.context.enable_nrt:
            resume_index_ptr = cgutils.gep(builder, genptr, 0, 0,
                                           name='gen.resume_index')
            resume_index = builder.load(resume_index_ptr)

            # If resume_index is 0, next() was never called
            # Note: The init func has to acquire the reference
            #       of all arguments; otherwise, they will be destroyed at the
            #       end of the init func.  The proper release of these
            #       references relies on the actual NPM function, which is
            #       never called if the generator exit before any entry into
            #       the NPM function.
            need_args_cleanup = builder.icmp_signed(
                '==', resume_index, Constant.int(resume_index.type, 0))

            with cgutils.ifthen(builder, need_args_cleanup):
                gen_args_ptr = cgutils.gep(builder, genptr, 0, 1, name="gen_args")
                assert len(self.fndesc.argtypes) == len(gen_args_ptr.type.pointee)
                for elem_idx, argty in enumerate(self.fndesc.argtypes):
                    argptr = cgutils.gep(builder, gen_args_ptr, 0, elem_idx)
                    argval = builder.load(argptr)
                    self.context.nrt_decref(builder, argty, argval)

            # Always run the finalizer to clear the block
            gen_state_ptr = cgutils.gep(builder, genptr, 0, 2, name='gen.state')

            for state_index in range(len(self.gentype.state_types)):
                state_slot = cgutils.gep(builder, gen_state_ptr,
                                         0, state_index)
                ty = self.gentype.state_types[state_index]
                val = self.context.unpack_value(builder, ty, state_slot)
                if self.context.enable_nrt:
                    self.context.nrt_decref(builder, ty, val)

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
            gen_func=self.interp.bytecode.func,
            yield_type=types.pyobject,
            arg_types=(types.pyobject,) * self.interp.arg_count,
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
        lower.builder.store(Constant.null(self.gen_state_ptr.type.pointee),
                            self.gen_state_ptr)

    def lower_finalize_func_body(self, builder, genptr):
        """
        Lower the body of the generator's finalizer: decref all live
        state variables.
        """
        pyapi = self.context.get_python_api(builder)
        resume_index_ptr = cgutils.gep(builder, genptr, 0, 0,
                                       name='gen.resume_index')
        resume_index = builder.load(resume_index_ptr)
        # If resume_index is 0, next() was never called
        # If resume_index is -1, generator terminated cleanly
        # (note function arguments are saved in state variables,
        #  so they don't need a separate cleanup step)
        need_cleanup = builder.icmp_signed(
            '>', resume_index, Constant.int(resume_index.type, 0))

        with cgutils.if_unlikely(builder, need_cleanup):
            # Decref all live vars (some may be NULL)
            gen_state_ptr = cgutils.gep(builder, genptr, 0, 2,
                                        name='gen.state')
            for state_index in range(len(self.gentype.state_types)):
                state_slot = cgutils.gep(builder, gen_state_ptr,
                                         0, state_index)
                ty = self.gentype.state_types[state_index]
                val = self.context.unpack_value(builder, ty, state_slot)
                pyapi.decref(val)

        builder.ret_void()


class LowerYield(object):
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
        self.live_var_indices = [lower.generator_info.state_vars.index(v)
                                 for v in live_vars]

    def lower_yield_suspend(self):
        # Save live vars in state
        for state_index, name in zip(self.live_var_indices, self.live_vars):
            state_slot = cgutils.gep(self.builder, self.gen_state_ptr,
                                     0, state_index)
            ty = self.gentype.state_types[state_index]
            val = self.lower.loadvar(name)
            self.context.pack_value(self.builder, ty, val, state_slot)
        # Save resume index
        indexval = Constant.int(self.resume_index_ptr.type.pointee,
                                self.inst.index)
        self.builder.store(indexval, self.resume_index_ptr)

    def lower_yield_resume(self):
        # Emit resumption point
        self.genlower.create_resumption_block(self.lower, self.inst.index)
        # Reload live vars from state
        for state_index, name in zip(self.live_var_indices, self.live_vars):
            state_slot = cgutils.gep(self.builder, self.gen_state_ptr,
                                     0, state_index)
            ty = self.gentype.state_types[state_index]
            val = self.context.unpack_value(self.builder, ty, state_slot)
            self.lower.storevar(val, name)
            # Previous storevar is making an extra incref
            if self.context.enable_nrt:
                self.lower.decref(ty, val)
