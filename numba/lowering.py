from __future__ import print_function, division, absolute_import

from collections import defaultdict
import sys
from types import ModuleType

from llvm.core import Type, Builder, Module
import llvm.core as lc

from numba import _dynfunc, ir, types, cgutils, utils, config, cffi_support, typing


class LoweringError(Exception):
    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(LoweringError, self).__init__("%s\n%s" % (msg, loc.strformat()))


class ForbiddenConstruct(LoweringError):
    pass


def default_mangler(name, argtypes):
    codedargs = '.'.join(str(a).replace(' ', '_') for a in argtypes)
    return '.'.join([name, codedargs])


# A dummy module for dynamically-generated functions
_dynamic_modname = '<dynamic>'
_dynamic_module = ModuleType(_dynamic_modname)


class FunctionDescriptor(object):
    __slots__ = ('native', 'modname', 'qualname', 'doc', 'blocks', 'typemap',
                 'calltypes', 'args', 'kws', 'restype', 'argtypes',
                 'mangled_name', 'unique_name', 'globals')

    def __init__(self, native, modname, qualname, unique_name, doc, blocks,
                 typemap, restype, calltypes, args, kws, mangler=None,
                 argtypes=None, globals=None):
        self.native = native
        self.modname = modname
        self.qualname = qualname
        self.unique_name = unique_name
        self.globals = globals if globals is not None else {}
        self.doc = doc
        self.blocks = blocks
        self.typemap = typemap
        self.calltypes = calltypes
        self.args = args
        self.kws = kws
        self.restype = restype
        # Argument types
        self.argtypes = argtypes or [self.typemap[a] for a in args]
        mangler = default_mangler if mangler is None else mangler
        self.mangled_name = mangler(self.qualname, self.argtypes)

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

    def __repr__(self):
        return "<function descriptor %r>" % (self.unique_name)

    @classmethod
    def _get_function_info(cls, interp):
        """
        Returns
        -------
        qualname, unique_name, modname, doc, args, kws, globals

        ``unique_name`` must be a unique name in ``pymod``.
        """
        func = interp.bytecode.func
        qualname = interp.bytecode.func_qualname
        modname = func.__module__
        doc = func.__doc__ or ''
        args = interp.argspec.args
        kws = ()        # TODO

        if modname is None:
            # For dynamically generated functions (e.g. compile()),
            # add the function id to the name to create a unique name.
            unique_name = "%s$%d" % (qualname, id(func))
            modname = _dynamic_modname
        else:
            # For a top-level function or closure, make sure to disambiguate
            # the function name.
            # TODO avoid unnecessary recompilation of the same function
            unique_name = "%s$%d" % (qualname, func.__code__.co_firstlineno)

        return qualname, unique_name, modname, doc, args, kws, func.__globals__

    @classmethod
    def _from_python_function(cls, interp, typemap, restype, calltypes,
                              native, mangler=None):
        (qualname, unique_name, modname, doc, args, kws, func_globals
         )= cls._get_function_info(interp)
        sortedblocks = utils.SortedMap(utils.dict_iteritems(interp.blocks))
        self = cls(native, modname, qualname, unique_name, doc,
                   sortedblocks, typemap, restype, calltypes,
                   args, kws, mangler=mangler, globals=func_globals)
        return self


class PythonFunctionDescriptor(FunctionDescriptor):
    __slots__ = ()

    @classmethod
    def from_specialized_function(cls, interp, typemap, restype, calltypes,
                                  mangler):
        """
        Build a FunctionDescriptor for a specialized Python function.
        """
        return cls._from_python_function(interp, typemap, restype, calltypes,
                                         native=True, mangler=mangler)

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
                blocks=None, typemap=None, restype=restype, calltypes=None,
                args=args, kws=None, mangler=lambda a, x: a, argtypes=argtypes,
                globals={})


class BaseLower(object):
    """
    Lower IR to LLVM
    """
    def __init__(self, context, fndesc):
        self.context = context
        self.fndesc = fndesc
        # Initialize LLVM
        self.module = Module.new("module.%s" % self.fndesc.unique_name)

        # Python execution environment (will be available to the compiled
        # function).
        self.env = _dynfunc.Environment(
            globals=self.fndesc.lookup_module().__dict__)

        # Setup function
        self.function = context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)

        # Internal states
        self.blkmap = {}
        self.varmap = {}
        self.firstblk = min(self.fndesc.blocks.keys())
        self.loc = -1

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

    def lower(self):
        # Init argument variables
        fnargs = self.context.get_arguments(self.function)
        for ak, av in zip(self.fndesc.args, fnargs):
            at = self.typeof(ak)
            av = self.context.get_argument_value(self.builder, at, av)
            av = self.init_argument(av)
            self.storevar(av, ak)

        # Init blocks
        for offset in self.fndesc.blocks:
            bname = "B%d" % offset
            self.blkmap[offset] = self.function.append_basic_block(bname)

        self.pre_lower()
        # pre_lower() may have changed the current basic block
        entry_block_tail = self.builder.basic_block

        # Lower all blocks
        for offset, block in self.fndesc.blocks.items():
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.lower_block(block)

        self.post_lower()

        # Close tail of entry block
        self.builder.position_at_end(entry_block_tail)
        self.builder.branch(self.blkmap[self.firstblk])

        if config.DUMP_LLVM:
            print(("LLVM DUMP %s" % self.fndesc).center(80, '-'))
            print(self.module)
            print('=' * 80)
        self.module.verify()
        # Run function-level optimize to reduce memory usage and improve
        # module-level optimization
        self.context.optimize_function(self.function)

        if config.NUMBA_DUMP_FUNC_OPT:
            print(("LLVM FUNCTION OPTIMIZED DUMP %s" %
                   self.fndesc).center(80, '-'))
            print(self.module)
            print('=' * 80)

    def init_argument(self, arg):
        return arg

    def lower_block(self, block):
        for inst in block.body:
            self.loc = inst.loc
            try:
                self.lower_inst(inst)
            except LoweringError:
                raise
            except Exception as e:
                msg = "Internal error:\n%s: %s" % (type(e).__name__, e)
                raise LoweringError(msg, inst.loc)

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
            val = self.loadvar(inst.value.name)
            oty = self.typeof(inst.value.name)
            ty = self.fndesc.restype
            if isinstance(ty, types.Optional):
                if oty == types.none:
                    self.context.return_native_none(self.builder)
                    return
                else:
                    ty = ty.type

            if ty != oty:
                val = self.context.cast(self.builder, val, oty, ty)
            retval = self.context.get_return_value(self.builder, ty, val)
            self.context.return_value(self.builder, retval)

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
            excid = self.context.add_exception(inst.exception)
            self.context.return_user_exc(self.builder, excid)

        else:
            raise NotImplementedError(type(inst))

    def lower_assign(self, ty, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            return self.context.get_constant_generic(self.builder, ty,
                                                     value.value)

        elif isinstance(value, ir.Expr):
            return self.lower_expr(ty, value)

        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            oty = self.typeof(value.name)
            return self.context.cast(self.builder, val, oty, ty)

        # In nopython mode, closure vars are frozen like globals
        elif isinstance(value, (ir.Global, ir.FreeVar)):
            if (isinstance(ty, types.Dummy) or
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

            argvals = [self.loadvar(a.name) for a in expr.args]
            argtyps = [self.typeof(a.name) for a in expr.args]
            signature = self.fndesc.calltypes[expr]

            if isinstance(expr.func, ir.Intrinsic):
                fnty = expr.func.name
                castvals = expr.func.args
            else:
                assert not expr.kws, expr.kws
                fnty = self.typeof(expr.func.name)

                castvals = [self.context.cast(self.builder, av, at, ft)
                            for av, at, ft in zip(argvals, argtyps,
                                                  signature.args)]

            if isinstance(fnty, types.Method):
                # Method of objects are handled differently
                fnobj = self.loadvar(expr.func.name)
                res = self.context.call_class_method(self.builder, fnobj,
                                                     signature, castvals)

            elif isinstance(fnty, types.FunctionPointer):
                # Handle function pointer
                pointer = fnty.funcptr
                res = self.context.call_function_pointer(self.builder, pointer,
                                                         signature, castvals)

            elif isinstance(fnty, cffi_support.ExternCFunction):
                # XXX unused?
                fndesc = ExternalFunctionDescriptor(
                    fnty.symbol, fnty.restype, fnty.argtypes)
                func = self.context.declare_external_function(
                        cgutils.get_module(self.builder), fndesc)
                res = self.context.call_external_function(self.builder, func, fndesc.argtypes, castvals)

            else:
                # Normal function resolution
                impl = self.context.get_function(fnty, signature)
                res = impl(self.builder, castvals)
                libs = getattr(impl, "libs", ())
                if libs:
                    self.context.add_libs(libs)
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'pair_first':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            item = self.context.pair_first(self.builder, val, ty)
            return self.context.get_argument_value(self.builder,
                                                   ty.first_type, item)

        elif expr.op == 'pair_second':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            item = self.context.pair_second(self.builder, val, ty)
            return self.context.get_argument_value(self.builder,
                                                   ty.second_type, item)

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
            excid = self.context.add_exception(ValueError)
            # We call iternext() as many times as desired (`expr.count`).
            for i in range(expr.count):
                pair = iternext_impl(self.builder, (iterobj,))
                is_valid = self.context.pair_second(self.builder,
                                                    pair, pairty)
                with cgutils.if_unlikely(self.builder,
                                         self.builder.not_(is_valid)):
                    self.context.return_user_exc(self.builder, excid)
                item = self.context.pair_first(self.builder,
                                               pair, pairty)
                tup = self.builder.insert_value(tup, item, i)

            # Call iternext() once more to check that the iterator
            # is exhausted.
            pair = iternext_impl(self.builder, (iterobj,))
            is_valid = self.context.pair_second(self.builder,
                                                pair, pairty)
            with cgutils.if_unlikely(self.builder, is_valid):
                self.context.return_user_exc(self.builder, excid)

            return tup

        elif expr.op == "getattr":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
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
