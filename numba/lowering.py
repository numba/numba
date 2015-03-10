from __future__ import print_function, division, absolute_import

from collections import defaultdict
import itertools
import sys
from types import ModuleType

from llvmlite.llvmpy.core import Constant, Type, Builder


from numba import (_dynfunc, ir, types, cgutils, utils, config,
                   cffi_support, typing, six)


class LoweringError(Exception):
    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(LoweringError, self).__init__("%s\n%s" % (msg, loc.strformat()))


class ForbiddenConstruct(LoweringError):
    pass


def transform_arg_name(arg):
    if isinstance(arg, types.Record):
        return "Record_%s" % arg._code
    elif (isinstance(arg, types.Array) and
          isinstance(arg.dtype, types.Record)):
        type_name = "array" if arg.mutable else "readonly array"
        return ("%s(Record_%s, %sd, %s)"
                % (type_name, arg.dtype._code, arg.ndim, arg.layout))
    else:
        return str(arg)


def default_mangler(name, argtypes):
    codedargs = '.'.join(transform_arg_name(a).replace(' ', '_')
                             for a in argtypes)
    return '.'.join([name, codedargs])


# A dummy module for dynamically-generated functions
_dynamic_modname = '<dynamic>'
_dynamic_module = ModuleType(_dynamic_modname)
_dynamic_module.__builtins__ = six.moves.builtins


class FunctionDescriptor(object):
    __slots__ = ('native', 'modname', 'qualname', 'doc', 'typemap',
                 'calltypes', 'args', 'kws', 'restype', 'argtypes',
                 'mangled_name', 'unique_name', 'inline')

    _unique_ids = itertools.count(1)

    def __init__(self, native, modname, qualname, unique_name, doc,
                 typemap, restype, calltypes, args, kws, mangler=None,
                 argtypes=None, inline=False):
        self.native = native
        self.modname = modname
        self.qualname = qualname
        self.unique_name = unique_name
        self.doc = doc
        self.typemap = typemap
        self.calltypes = calltypes
        self.args = args
        self.kws = kws
        self.restype = restype
        # Argument types
        self.argtypes = argtypes or [self.typemap[a] for a in args]
        mangler = default_mangler if mangler is None else mangler
        # The mangled name *must* be unique, else the wrong function can
        # be chosen at link time.
        if self.modname:
            self.mangled_name = mangler('%s.%s' % (self.modname, self.unique_name),
                                        self.argtypes)
        else:
            self.mangled_name = mangler(self.unique_name, self.argtypes)
        self.inline = inline

    def lookup_module(self):
        """
        Return the module in which this function is supposed to exist.
        This may be a dummy module if the function was dynamically
        generated.
        """
        if self.modname == _dynamic_modname:
            return _dynamic_module
        else:
            return sys.modules[self.modname]

    @property
    def llvm_func_name(self):
        return self.mangled_name

    @property
    def llvm_cpython_wrapper_name(self):
        return 'wrapper.' + self.mangled_name

    def __repr__(self):
        return "<function descriptor %r>" % (self.unique_name)

    @classmethod
    def _get_function_info(cls, interp):
        """
        Returns
        -------
        qualname, unique_name, modname, doc, args, kws, globals

        ``unique_name`` must be a unique name.
        """
        func = interp.bytecode.func
        qualname = interp.bytecode.func_qualname
        modname = func.__module__
        doc = func.__doc__ or ''
        args = interp.argspec.args
        kws = ()        # TODO

        if modname is None:
            # Dynamically generated function.
            modname = _dynamic_modname

        # Even the same function definition can be compiled into
        # several different function objects with distinct closure
        # variables, so we make sure to disambiguish using an unique id.
        unique_name = "%s$%d" % (qualname, next(cls._unique_ids))

        return qualname, unique_name, modname, doc, args, kws

    @classmethod
    def _from_python_function(cls, interp, typemap, restype, calltypes,
                              native, mangler=None, inline=False):
        (qualname, unique_name, modname, doc, args, kws,
         )= cls._get_function_info(interp)
        self = cls(native, modname, qualname, unique_name, doc,
                   typemap, restype, calltypes,
                   args, kws, mangler=mangler, inline=inline)
        return self


class PythonFunctionDescriptor(FunctionDescriptor):
    __slots__ = ()

    @classmethod
    def from_specialized_function(cls, interp, typemap, restype, calltypes,
                                  mangler, inline):
        """
        Build a FunctionDescriptor for a specialized Python function.
        """
        return cls._from_python_function(interp, typemap, restype, calltypes,
                                         native=True, mangler=mangler,
                                         inline=inline)

    @classmethod
    def from_object_mode_function(cls, interp):
        """
        Build a FunctionDescriptor for a Python function to be compiled
        and executed in object mode.
        """
        typemap = defaultdict(lambda: types.pyobject)
        calltypes = typemap.copy()
        restype = types.pyobject
        return cls._from_python_function(interp, typemap, restype, calltypes,
                                         native=False)


class GeneratorDescriptor(FunctionDescriptor):
    __slots__ = ()

    @classmethod
    def from_generator_fndesc(cls, interp, fndesc, mangler):
        """
        Build a GeneratorDescriptor for the generator returned by the
        function described by *fndesc*.
        """
        gentype = fndesc.restype
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


class ExternalFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for opaque external functions
    (e.g. raw C functions).
    """
    __slots__ = ()

    def __init__(self, name, restype, argtypes):
        args = ["arg%d" % i for i in range(len(argtypes))]
        super(ExternalFunctionDescriptor, self).__init__(native=True,
                modname=None, qualname=name, unique_name=name, doc='',
                typemap=None, restype=restype, calltypes=None,
                args=args, kws=None, mangler=lambda a, x: a,
                argtypes=argtypes)


class BaseLower(object):
    """
    Lower IR to LLVM
    """
    def __init__(self, context, library, fndesc, interp):
        self.context = context
        self.library = library
        self.fndesc = fndesc
        self.gendesc = None
        self.blocks = utils.SortedMap(utils.iteritems(interp.blocks))
        self.interp = interp
        self.call_conv = context.call_conv
        self.generator_info = self.interp.generator_info

        # Initialize LLVM
        self.module = self.library.create_ir_module(self.fndesc.unique_name)

        # Python execution environment (will be available to the compiled
        # function).
        self.env = _dynfunc.Environment(
            globals=self.fndesc.lookup_module().__dict__)

        # Internal states
        self.blkmap = {}
        self.varmap = {}
        self.firstblk = min(self.blocks.keys())
        self.loc = -1
        self.resume_blocks = {}

        # Subclass initialization
        self.init()

    def init(self):
        pass

    def pre_lower(self):
        """
        Called before lowering all blocks.
        """

    def post_lower(self):
        """
        Called after all blocks are lowered
        """

    def pre_block(self, block):
        """
        Called before lowering a block.
        """

    def return_exception(self, exc_class, exc_args=None):
        self.call_conv.return_user_exc(self.builder, exc_class, exc_args)

    def lower(self):
        if self.generator_info is None:
            self._lower_normal_function(self.fndesc)
        else:
            self.gentype = self.fndesc.restype
            assert isinstance(self.gentype, types.Generator)
            self._lower_generator_init()
            self._lower_generator_next()

        if config.DUMP_LLVM:
            print(("LLVM DUMP %s" % self.fndesc).center(80, '-'))
            print(self.module)
            print('=' * 80)

        # Materialize LLVM Module
        self.library.add_ir_module(self.module)

    def _lower_normal_function(self, fndesc):
        self.setup_function(fndesc)

        # Init argument values
        rawfnargs = self.call_conv.get_arguments(self.function)
        arginfo = self.context.get_arg_packer(fndesc.argtypes)
        self.fnargs = arginfo.from_arguments(self.builder, rawfnargs)

        # Init blocks
        for offset in self.blocks:
            bname = "B%s" % offset
            self.blkmap[offset] = self.function.append_basic_block(bname)

        self.pre_lower()
        # pre_lower() may have changed the current basic block
        entry_block_tail = self.builder.basic_block

        # Lower all blocks
        for offset, block in self.blocks.items():
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.lower_block(block)

        self.post_lower()

        # Close tail of entry block
        self.builder.position_at_end(entry_block_tail)
        self.builder.branch(self.blkmap[self.firstblk])

        # Run target specific post lowering transformation
        self.context.post_lowering(self.function)

    def _lower_generator_init(self):
        self.setup_function(self.fndesc)

        # Init argument values
        rawfnargs = self.call_conv.get_arguments(self.function)
        arginfo = self.context.get_arg_packer(self.fndesc.argtypes)
        self.fnargs = arginfo.from_arguments(self.builder, rawfnargs)

        self.pre_lower()

        model = self.context.data_model_manager[self.gentype]

        retty = self.context.get_return_type(self.gentype)
        argsty = retty.elements[1]
        argsval = cgutils.make_anonymous_struct(self.builder, self.fnargs,
                                                argsty)
        resume_index = self.context.get_constant(types.int32, 0)
        retval = cgutils.make_anonymous_struct(self.builder, [resume_index, argsval],
                                               retty)
        self.call_conv.return_value(self.builder, retval)

        self.post_lower()

    def _lower_generator_next(self):
        self.gendesc = GeneratorDescriptor.from_generator_fndesc(
            self.interp, self.fndesc, self.context.mangler)
        self.setup_function(self.gendesc)

        assert self.gendesc.argtypes[0] == self.gentype

        # Extract argument values and other information from generator struct
        genptr, = self.call_conv.get_arguments(self.function)
        for i, ty in enumerate(self.gentype.arg_types):
            argptr = cgutils.gep(self.builder, genptr, 0, 1, i)
            self.fnargs[i] = self.context.unpack_value(self.builder, ty, argptr)
        self.resume_index_ptr = cgutils.gep(self.builder, genptr, 0, 0,
                                            name='gen.resume_index')
        self.gen_state_ptr = cgutils.gep(self.builder, genptr, 0, 2,
                                         name='gen.state')

        prologue = self.function.append_basic_block("generator_prologue")

        # Init Python blocks
        for offset in self.blocks:
            bname = "B%s" % offset
            self.blkmap[offset] = self.function.append_basic_block(bname)

        self.pre_lower()

        # pre_lower() may have changed the current basic block
        entry_block_tail = self.builder.basic_block

        # Lower all Python blocks
        for offset, block in self.blocks.items():
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.lower_block(block)

        self.post_lower()

        # Add block for StopIteration on entry
        stop_block = self.function.append_basic_block("stop_iteration")
        self.builder.position_at_end(stop_block)
        self.call_conv.return_stop_iteration(self.builder)

        # Add prologue switch to resume blocks
        self.builder.position_at_end(prologue)
        # First Python block is also the resume point on first next() call
        first_block = self.resume_blocks[0] = self.blkmap[self.firstblk]

        # Create resume points
        switch = self.builder.switch(self.builder.load(self.resume_index_ptr),
                                     stop_block)
        for index, block in self.resume_blocks.items():
            switch.add_case(index, block)

        # Close tail of entry block
        self.builder.position_at_end(entry_block_tail)
        self.builder.branch(prologue)

        # Run target specific post lowering transformation
        self.context.post_lowering(self.function)

        self.context.insert_generator(self.gentype, self.gendesc, [self.library])

    def create_cpython_wrapper(self, release_gil=False):
        """
        Create CPython wrapper(s).
        """
        if self.gendesc:
            self.context.create_cpython_wrapper(self.library, self.gendesc,
                                                self.call_helper,
                                                release_gil=release_gil)
        self.context.create_cpython_wrapper(self.library, self.fndesc,
                                            self.call_helper,
                                            release_gil=release_gil)

    def lower_block(self, block):
        self.pre_block(block)
        for inst in block.body:
            self.loc = inst.loc
            try:
                self.lower_inst(inst)
            except LoweringError:
                raise
            except Exception as e:
                msg = "Internal error:\n%s: %s" % (type(e).__name__, e)
                raise LoweringError(msg, inst.loc)

    def setup_function(self, fndesc):
        # Setup function
        self.function = self.context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)
        self.call_helper = self.call_conv.init_call_helper(self.builder)

    def typeof(self, varname):
        return self.fndesc.typemap[varname]


class Lower(BaseLower):

    def lower_inst(self, inst):
        if config.DEBUG_JIT:
            self.context.debug_print(self.builder, str(inst))
        if isinstance(inst, ir.Assign):
            ty = self.typeof(inst.target.name)
            val = self.lower_assign(ty, inst)
            self.storevar(val, inst.target.name)

        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]

            condty = self.typeof(inst.cond.name)
            pred = self.context.cast(self.builder, cond, condty, types.boolean)
            assert pred.type == Type.int(1), ("cond is not i1: %s" % pred.type)
            self.builder.cbranch(pred, tr, fl)

        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)

        elif isinstance(inst, ir.Return):
            if self.generator_info:
                # StopIteration
                indexval = Constant.int(self.resume_index_ptr.type.pointee, -1)
                self.builder.store(indexval, self.resume_index_ptr)
                self.call_conv.return_stop_iteration(self.builder)
                return
            val = self.loadvar(inst.value.name)
            oty = self.typeof(inst.value.name)
            ty = self.fndesc.restype
            if isinstance(ty, types.Optional):
                # If returning an optional type
                self.call_conv.return_optional_value(self.builder, ty, oty, val)
                return
            if ty != oty:
                val = self.context.cast(self.builder, val, oty, ty)
            retval = self.context.get_return_value(self.builder, ty, val)
            self.call_conv.return_value(self.builder, retval)

        elif isinstance(inst, ir.SetItem):
            target = self.loadvar(inst.target.name)
            value = self.loadvar(inst.value.name)
            index = self.loadvar(inst.index.name)

            targetty = self.typeof(inst.target.name)
            valuety = self.typeof(inst.value.name)
            indexty = self.typeof(inst.index.name)

            signature = self.fndesc.calltypes[inst]
            assert signature is not None
            impl = self.context.get_function('setitem', signature)

            # Convert argument to match
            if isinstance(targetty, types.Optional):
                target = self.context.cast(self.builder, target, targetty,
                                           targetty.type)
            else:
                assert targetty == signature.args[0]

            index = self.context.cast(self.builder, index, indexty,
                                      signature.args[1])
            value = self.context.cast(self.builder, value, valuety,
                                      signature.args[2])

            return impl(self.builder, (target, index, value))

        elif isinstance(inst, ir.Del):
            pass

        elif isinstance(inst, ir.SetAttr):
            target = self.loadvar(inst.target.name)
            value = self.loadvar(inst.value.name)
            signature = self.fndesc.calltypes[inst]

            targetty = self.typeof(inst.target.name)
            valuety = self.typeof(inst.value.name)
            assert signature is not None
            assert signature.args[0] == targetty
            impl = self.context.get_setattr(inst.attr, signature)

            # Convert argument to match
            value = self.context.cast(self.builder, value, valuety,
                                      signature.args[1])

            return impl(self.builder, (target, value))

        elif isinstance(inst, ir.Raise):
            self.lower_raise(inst)

        else:
            raise NotImplementedError(type(inst))

    def lower_raise(self, inst):
        if inst.exception is None:
            # Reraise
            self.return_exception(None)
        else:
            exctype = self.typeof(inst.exception.name)
            if isinstance(exctype, types.ExceptionInstance):
                # raise <instance> => find the instantiation site
                excdef = self.interp.get_definition(inst.exception)
                if (not isinstance(excdef, ir.Expr) or excdef.op != 'call'
                    or excdef.kws):
                    raise NotImplementedError("unsupported kind of raising")
                # Try to infer the args tuple
                args = tuple(self.interp.get_definition(arg).infer_constant()
                             for arg in excdef.args)
            elif isinstance(exctype, types.ExceptionType):
                args = None
            else:
                raise NotImplementedError("cannot raise value of type %s"
                                          % (exctype,))
            self.return_exception(exctype.exc_class, args)

    def lower_assign(self, ty, inst):
        value = inst.value
        # In nopython mode, closure vars are frozen like globals
        if isinstance(value, (ir.Const, ir.Global, ir.FreeVar)):
            if isinstance(ty, types.ExternalFunctionPointer):
                return self.context.get_constant_generic(self.builder, ty,
                                                         value.value)

            elif (isinstance(ty, types.Dummy) or
                    isinstance(ty, types.Module) or
                    isinstance(ty, types.Function) or
                    isinstance(ty, types.Dispatcher)):
                return self.context.get_dummy_value()

            elif isinstance(ty, types.Array):
                return self.context.make_constant_array(self.builder, ty,
                                                        value.value)

            else:
                return self.context.get_constant_generic(self.builder, ty,
                                                         value.value)

        elif isinstance(value, ir.Expr):
            return self.lower_expr(ty, value)

        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            oty = self.typeof(value.name)
            return self.context.cast(self.builder, val, oty, ty)

        elif isinstance(value, ir.Arg):
            return self.fnargs[value.index]

        elif isinstance(value, ir.Yield):
            return self.lower_yield(ty, value)

        else:
            raise NotImplementedError(type(value), value)

    def lower_binop(self, resty, expr):
        lhs = expr.lhs
        rhs = expr.rhs
        lty = self.typeof(lhs.name)
        rty = self.typeof(rhs.name)
        lhs = self.loadvar(lhs.name)
        rhs = self.loadvar(rhs.name)
        # Get function
        signature = self.fndesc.calltypes[expr]
        impl = self.context.get_function(expr.fn, signature)
        # Convert argument to match
        lhs = self.context.cast(self.builder, lhs, lty, signature.args[0])
        rhs = self.context.cast(self.builder, rhs, rty, signature.args[1])
        res = impl(self.builder, (lhs, rhs))
        return self.context.cast(self.builder, res, signature.return_type,
                                 resty)

    def lower_yield(self, ty, inst):
        block, _i = self.generator_info.yield_points[inst.index]
        assert _i is inst
        # Save live vars in state
        live_vars = self.interp.get_block_entry_vars(block)
        indices = [self.generator_info.state_vars.index(v) for v in live_vars]
        for state_index, name in zip(indices, live_vars):
            state_slot = cgutils.gep(self.builder, self.gen_state_ptr,
                                     0, state_index)
            ty = self.gentype.state_types[state_index]
            self.context.pack_value(self.builder, ty, self.loadvar(name), state_slot)
        # Save resume index
        indexval = Constant.int(self.resume_index_ptr.type.pointee, inst.index)
        self.builder.store(indexval, self.resume_index_ptr)
        # Yield to caller
        self.call_conv.return_value(self.builder, self.loadvar(inst.value.name))
        # Emit resumption point
        block_name = "generator_resume%d" % (inst.index)
        block = self.function.append_basic_block(block_name)
        self.builder.position_at_end(block)
        self.resume_blocks[inst.index] = block
        # Reload live vars from state
        for state_index, name in zip(indices, live_vars):
            state_slot = cgutils.gep(self.builder, self.gen_state_ptr,
                                     0, state_index)
            ty = self.gentype.state_types[state_index]
            val = self.context.unpack_value(self.builder, ty, state_slot)
            self.storevar(val, name)
        return self.context.get_constant_generic(self.builder, ty, None)

    def lower_expr(self, resty, expr):
        if expr.op == 'binop':
            return self.lower_binop(resty, expr)
        elif expr.op == 'inplace_binop':
            lty = self.typeof(expr.lhs.name)
            if not lty.mutable:
                # inplace operators on non-mutable types reuse the same
                # definition as the corresponding copying operators.
                return self.lower_binop(resty, expr)
        elif expr.op == 'unary':
            val = self.loadvar(expr.value.name)
            typ = self.typeof(expr.value.name)
            # Get function
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.fn, signature)
            # Convert argument to match
            val = self.context.cast(self.builder, val, typ, signature.args[0])
            res = impl(self.builder, [val])
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'call':
            signature = self.fndesc.calltypes[expr]

            if isinstance(expr.func, ir.Intrinsic):
                fnty = expr.func.name
                castvals = expr.func.args
            else:
                fnty = self.typeof(expr.func.name)
                if expr.kws:
                    # Fold keyword arguments
                    try:
                        pysig = fnty.pysig
                    except AttributeError:
                        raise NotImplementedError("unsupported keyword arguments "
                                                  "when calling %s" % (fnty,))
                    ba = pysig.bind(*expr.args, **dict(expr.kws))
                    assert not ba.kwargs
                    args = ba.args
                else:
                    args = expr.args

                argvals = [self.loadvar(a.name) for a in args]
                argtyps = [self.typeof(a.name) for a in args]

                castvals = [self.context.cast(self.builder, av, at, ft)
                            for av, at, ft in zip(argvals, argtyps,
                                                  signature.args)]

            if isinstance(fnty, types.ExternalFunction):
                # Handle a named external function
                fndesc = ExternalFunctionDescriptor(
                    fnty.symbol, fnty.sig.return_type, fnty.sig.args)
                func = self.context.declare_external_function(
                        cgutils.get_module(self.builder), fndesc)
                res = self.context.call_external_function(
                    self.builder, func, fndesc.argtypes, castvals)

            elif isinstance(fnty, types.Method):
                # Method of objects are handled differently
                fnobj = self.loadvar(expr.func.name)
                res = self.context.call_class_method(self.builder, fnobj,
                                                     signature, castvals)

            elif isinstance(fnty, types.ExternalFunctionPointer):
                # Handle a C function pointer
                pointer = self.loadvar(expr.func.name)
                res = self.context.call_function_pointer(self.builder, pointer,
                                                         castvals, fnty.cconv)

            else:
                if isinstance(signature.return_type, types.Phantom):
                    return self.context.get_dummy_value()
                # Normal function resolution (for Numba-compiled functions)
                impl = self.context.get_function(fnty, signature)
                if signature.recvr:
                    # The "self" object is passed as the function object
                    # for bounded function
                    the_self = self.loadvar(expr.func.name)
                    # Prepend the self reference
                    castvals = [the_self] + castvals

                res = impl(self.builder, castvals)
                libs = getattr(impl, "libs", ())
                for lib in libs:
                    self.library.add_linking_library(lib)
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'pair_first':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            return self.context.pair_first(self.builder, val, ty)

        elif expr.op == 'pair_second':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            return self.context.pair_second(self.builder, val, ty)

        elif expr.op in ('getiter', 'iternext'):
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.op, signature)
            [fty] = signature.args
            castval = self.context.cast(self.builder, val, ty, fty)
            res = impl(self.builder, (castval,))
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'exhaust_iter':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            # If we have a heterogenous tuple, we needn't do anything,
            # and we can't iterate over it anyway.
            if isinstance(ty, types.Tuple):
                return val

            itemty = ty.iterator_type.yield_type
            tup = self.context.get_constant_undef(resty)
            pairty = types.Pair(itemty, types.boolean)
            getiter_sig = typing.signature(ty.iterator_type, ty)
            getiter_impl = self.context.get_function('getiter',
                                                     getiter_sig)
            iternext_sig = typing.signature(pairty, ty.iterator_type)
            iternext_impl = self.context.get_function('iternext',
                                                      iternext_sig)
            iterobj = getiter_impl(self.builder, (val,))
            # We call iternext() as many times as desired (`expr.count`).
            for i in range(expr.count):
                pair = iternext_impl(self.builder, (iterobj,))
                is_valid = self.context.pair_second(self.builder,
                                                    pair, pairty)
                with cgutils.if_unlikely(self.builder,
                                         self.builder.not_(is_valid)):
                    self.return_exception(ValueError)
                item = self.context.pair_first(self.builder,
                                               pair, pairty)
                tup = self.builder.insert_value(tup, item, i)

            # Call iternext() once more to check that the iterator
            # is exhausted.
            pair = iternext_impl(self.builder, (iterobj,))
            is_valid = self.context.pair_second(self.builder,
                                                pair, pairty)
            with cgutils.if_unlikely(self.builder, is_valid):
                self.return_exception(ValueError)

            return tup

        elif expr.op == "getattr":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)

            if isinstance(resty, types.BoundFunction):
                # if we are getting out a method, assume we have typed this
                # properly and just build a bound function object
                res = self.context.get_bound_function(self.builder, val, ty)
            else:
                impl = self.context.get_attribute(val, ty, expr.attr)

                if impl is None:
                    # ignore the attribute
                    res = self.context.get_dummy_value()
                else:
                    res = impl(self.context, self.builder, ty, val, expr.attr)
            return res

        elif expr.op == "static_getitem":
            baseval = self.loadvar(expr.value.name)
            indexval = self.context.get_constant(types.intp, expr.index)
            if cgutils.is_struct(baseval.type):
                # Statically extract the given element from the structure
                # (structures aren't dynamically indexable).
                return self.builder.extract_value(baseval, expr.index)
            else:
                # Fall back on the generic getitem() implementation
                # for this type.
                signature = typing.signature(resty,
                                             self.typeof(expr.value.name),
                                             types.intp)
                impl = self.context.get_function("getitem", signature)
                argvals = (baseval, indexval)
                res = impl(self.builder, argvals)
                return self.context.cast(self.builder, res, signature.return_type,
                                         resty)

        elif expr.op == "getitem":
            baseval = self.loadvar(expr.value.name)
            indexval = self.loadvar(expr.index.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function("getitem", signature)
            argvals = (baseval, indexval)
            argtyps = (self.typeof(expr.value.name),
                       self.typeof(expr.index.name))
            castvals = [self.context.cast(self.builder, av, at, ft)
                        for av, at, ft in zip(argvals, argtyps,
                                              signature.args)]
            res = impl(self.builder, castvals)
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == "build_tuple":
            itemvals = [self.loadvar(i.name) for i in expr.items]
            itemtys = [self.typeof(i.name) for i in expr.items]
            castvals = [self.context.cast(self.builder, val, fromty, toty)
                        for val, toty, fromty in zip(itemvals, resty, itemtys)]
            tup = self.context.get_constant_undef(resty)
            for i in range(len(castvals)):
                tup = self.builder.insert_value(tup, castvals[i], i)
            return tup

        elif expr.op == "cast":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            castval = self.context.cast(self.builder, val, ty, resty)
            return castval

        raise NotImplementedError(expr)

    def getvar(self, name):
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        if name not in self.varmap:
            self.varmap[name] = self.alloca_lltype(name, value.type)
        ptr = self.getvar(name)
        assert value.type == ptr.type.pointee,\
            "store %s to ptr of %s" % (value.type, ptr.type.pointee)
        self.builder.store(value, ptr)

    def alloca(self, name, type):
        lltype = self.context.get_value_type(type)
        return self.alloca_lltype(name, lltype)

    def alloca_lltype(self, name, lltype):
        return cgutils.alloca_once(self.builder, lltype, name=name)
