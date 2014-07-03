from __future__ import print_function, division, absolute_import

from collections import defaultdict
import sys
from types import ModuleType

from llvm.core import Type, Builder, Module
import llvm.core as lc

from numba import ir, types, cgutils, utils, config, cffi_support, typing


try:
    import builtins
except ImportError:
    import __builtin__ as builtins


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

# Issue #475: locals() is unsupported as calling it naively would give
# out wrong results.
_unsupported_builtins = set([locals])


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
        qualname = getattr(func, '__qualname__', interp.bytecode.func_name)
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

        # Install metadata
        md_pymod = cgutils.MetadataKeyStore(self.module, "python.module")
        md_pymod.set(fndesc.modname)

        # Setup function
        self.function = context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)
        # self.builder = cgutils.VerboseProxy(self.builder)

        # Internal states
        self.blkmap = {}
        self.varmap = {}
        self.firstblk = min(self.fndesc.blocks.keys())
        self.loc = -1

        # Subclass initialization
        self.init()

    def init(self):
        pass

    def post_lower(self):
        """Called after all blocks are lowered
        """
        pass

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
        # Lower all blocks
        for offset, block in self.fndesc.blocks.items():
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.lower_block(block)

        self.post_lower()
        # Close entry block
        self.builder.position_at_end(self.entry_block)
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

        elif isinstance(value, ir.Global):
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

    def lower_expr(self, resty, expr):
        if expr.op == 'binop':
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
        bb = self.builder.basic_block
        self.builder.position_at_end(self.entry_block)
        ptr = self.builder.alloca(lltype, name=name)
        self.builder.position_at_end(bb)
        return ptr


PYTHON_OPMAP = {
     '+': "number_add",
     '-': "number_subtract",
     '*': "number_multiply",
    '/?': "number_divide",
     '/': "number_truedivide",
    '//': "number_floordivide",
     '%': "number_remainder",
    '**': "number_power",
    '<<': "number_lshift",
    '>>': "number_rshift",
     '&': "number_and",
     '|': "number_or",
     '^': "number_xor",
}


class PyLower(BaseLower):
    def init(self):
        self.pyapi = self.context.get_python_api(self.builder)

        # Add error handling block
        self.ehblock = self.function.append_basic_block('error')

    def post_lower(self):
        with cgutils.goto_block(self.builder, self.ehblock):
            self.cleanup()
            self.context.return_exc(self.builder)

    def init_argument(self, arg):
        self.incref(arg)
        return arg

    def lower_inst(self, inst):
        if isinstance(inst, ir.Assign):
            value = self.lower_assign(inst)
            self.storevar(value, inst.target.name)

        elif isinstance(inst, ir.SetItem):
            target = self.loadvar(inst.target.name)
            index = self.loadvar(inst.index.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.object_setitem(target, index, value)
            self.check_int_status(ok)

        elif isinstance(inst, ir.StoreMap):
            dct = self.loadvar(inst.dct)
            key = self.loadvar(inst.key)
            value = self.loadvar(inst.value)
            ok = self.pyapi.dict_setitem(dct, key, value)
            self.check_int_status(ok)

        elif isinstance(inst, ir.Return):
            retval = self.loadvar(inst.value.name)
            self.incref(retval)
            self.cleanup()
            self.context.return_value(self.builder, retval)

        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            if cond.type == Type.int(1):
                istrue = cond
            else:
                istrue = self.pyapi.object_istrue(cond)
            zero = lc.Constant.null(istrue.type)
            pred = self.builder.icmp(lc.ICMP_NE, istrue, zero)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]
            self.builder.cbranch(pred, tr, fl)

        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)

        elif isinstance(inst, ir.Del):
            self.delvar(inst.value)

        elif isinstance(inst, ir.Raise):
            self.pyapi.raise_exception(inst.exception, inst.exception)
            self.return_exception_raised()

        else:
            raise NotImplementedError(type(inst), inst)

    def lower_assign(self, inst):
        """
        The returned object must have a new reference
        """
        value = inst.value
        if isinstance(value, ir.Const):
            return self.lower_const(value.value)
        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            self.incref(val)
            return val
        elif isinstance(value, ir.Expr):
            return self.lower_expr(value)
        elif isinstance(value, ir.Global):
            return self.lower_global(value.name, value.value)
        else:
            raise NotImplementedError(type(value), value)

    def lower_expr(self, expr):
        if expr.op == 'binop':
            lhs = self.loadvar(expr.lhs.name)
            rhs = self.loadvar(expr.rhs.name)
            if expr.fn in PYTHON_OPMAP:
                fname = PYTHON_OPMAP[expr.fn]
                fn = getattr(self.pyapi, fname)
                res = fn(lhs, rhs)
            else:
                # Assume to be rich comparision
                res = self.pyapi.object_richcompare(lhs, rhs, expr.fn)
            self.check_error(res)
            return res
        elif expr.op == 'unary':
            value = self.loadvar(expr.value.name)
            if expr.fn == '-':
                res = self.pyapi.number_negative(value)
            elif expr.fn == '+':
                res = self.pyapi.number_positive(value)
            elif expr.fn == 'not':
                res = self.pyapi.object_not(value)
                self.check_int_status(res)

                longval = self.builder.zext(res, self.pyapi.long)
                res = self.pyapi.bool_from_long(longval)
            elif expr.fn == '~':
                res = self.pyapi.number_invert(value)
            else:
                raise NotImplementedError(expr)
            self.check_error(res)
            return res
        elif expr.op == 'call':
            argvals = [self.loadvar(a.name) for a in expr.args]
            fn = self.loadvar(expr.func.name)
            if not expr.kws:
                # No keyword
                ret = self.pyapi.call_function_objargs(fn, argvals)
            else:
                # Have Keywords
                keyvalues = [(k, self.loadvar(v.name)) for k, v in expr.kws]
                args = self.pyapi.tuple_pack(argvals)
                kws = self.pyapi.dict_pack(keyvalues)
                ret = self.pyapi.call(fn, args, kws)
                self.decref(kws)
                self.decref(args)
            self.check_error(ret)
            return ret
        elif expr.op == 'getattr':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getattr_string(obj, expr.attr)
            self.check_error(res)
            return res
        elif expr.op == 'build_tuple':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.tuple_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'build_list':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.list_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'build_map':
            res = self.pyapi.dict_new(expr.size)
            self.check_error(res)
            return res
        elif expr.op == 'build_set':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.set_new()
            self.check_error(res)
            for it in items:
                ok = self.pyapi.set_add(res, it)
                self.check_int_status(ok)
            return res
        elif expr.op == 'getiter':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getiter(obj)
            self.check_error(res)
            return res
        elif expr.op == 'iternext':
            iterobj = self.loadvar(expr.value.name)
            item = self.pyapi.iter_next(iterobj)
            is_valid = cgutils.is_not_null(self.builder, item)
            pair = self.pyapi.tuple_new(2)
            with cgutils.ifelse(self.builder, is_valid) as (then, otherwise):
                with then:
                    self.pyapi.tuple_setitem(pair, 0, item)
                with otherwise:
                    self.check_occurred()
                    # Make the tuple valid by inserting None as dummy
                    # iteration "result" (it will be ignored).
                    self.pyapi.tuple_setitem(pair, 0, self.pyapi.make_none())
            self.pyapi.tuple_setitem(pair, 1, self.pyapi.bool_from_bool(is_valid))
            return pair
        elif expr.op == 'pair_first':
            pair = self.loadvar(expr.value.name)
            first = self.pyapi.tuple_getitem(pair, 0)
            self.incref(first)
            return first
        elif expr.op == 'pair_second':
            pair = self.loadvar(expr.value.name)
            second = self.pyapi.tuple_getitem(pair, 1)
            self.incref(second)
            return second
        elif expr.op == 'exhaust_iter':
            iterobj = self.loadvar(expr.value.name)
            tup = self.pyapi.sequence_tuple(iterobj)
            self.check_error(tup)
            # Check tuple size is as expected
            tup_size = self.pyapi.tuple_size(tup)
            expected_size = self.context.get_constant(types.intp, expr.count)
            has_wrong_size = self.builder.icmp(lc.ICMP_NE,
                                               tup_size, expected_size)
            with cgutils.if_unlikely(self.builder, has_wrong_size):
                excid = self.context.add_exception(ValueError)
                self.context.return_user_exc(self.builder, excid)
            return tup
        elif expr.op == 'getitem':
            value = self.loadvar(expr.value.name)
            index = self.loadvar(expr.index.name)
            res = self.pyapi.object_getitem(value, index)
            self.check_error(res)
            return res
        elif expr.op == 'static_getitem':
            value = self.loadvar(expr.value.name)
            index = self.context.get_constant(types.intp, expr.index)
            indexobj = self.pyapi.long_from_ssize_t(index)
            self.check_error(indexobj)
            res = self.pyapi.object_getitem(value, indexobj)
            self.decref(indexobj)
            self.check_error(res)
            return res
        elif expr.op == 'getslice':
            target = self.loadvar(expr.target.name)
            start = self.loadvar(expr.start.name)
            stop = self.loadvar(expr.stop.name)

            slicefn = self.get_builtin_obj("slice")
            sliceobj = self.pyapi.call_function_objargs(slicefn, (start, stop))
            self.decref(slicefn)
            self.check_error(sliceobj)

            res = self.pyapi.object_getitem(target, sliceobj)
            self.check_error(res)

            return res
        else:
            raise NotImplementedError(expr)

    def lower_const(self, const):
        if isinstance(const, str):
            ret = self.pyapi.string_from_constant_string(const)
            self.check_error(ret)
            return ret
        elif isinstance(const, complex):
            real = self.context.get_constant(types.float64, const.real)
            imag = self.context.get_constant(types.float64, const.imag)
            ret = self.pyapi.complex_from_doubles(real, imag)
            self.check_error(ret)
            return ret
        elif isinstance(const, float):
            fval = self.context.get_constant(types.float64, const)
            ret = self.pyapi.float_from_double(fval)
            self.check_error(ret)
            return ret
        elif isinstance(const, utils.INT_TYPES):
            if utils.bit_length(const) >= 64:
                raise ValueError("Integer is too big to be lowered")
            ival = self.context.get_constant(types.intp, const)
            return self.pyapi.long_from_ssize_t(ival)
        elif isinstance(const, tuple):
            items = [self.lower_const(i) for i in const]
            return self.pyapi.tuple_pack(items)
        elif const is Ellipsis:
            return self.get_builtin_obj("Ellipsis")
        elif const is None:
            return self.pyapi.make_none()
        else:
            raise NotImplementedError(type(const))

    def lower_global(self, name, value):
        """
        1) Check global scope dictionary.
        2) Check __builtins__.
            2a) is it a dictionary (for non __main__ module)
            2b) is it a module (for __main__ module)
        """
        moddict = self.pyapi.get_module_dict()
        obj = self.pyapi.dict_getitem_string(moddict, name)
        self.incref(obj)  # obj is borrowed

        if value in _unsupported_builtins:
            raise ForbiddenConstruct("builtins %s() is not supported"
                                     % name, loc=self.loc)

        if hasattr(builtins, name):
            obj_is_null = self.is_null(obj)
            bbelse = self.builder.basic_block

            with cgutils.ifthen(self.builder, obj_is_null):
                mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
                builtin = self.builtin_lookup(mod, name)
                bbif = self.builder.basic_block

            retval = self.builder.phi(self.pyapi.pyobj)
            retval.add_incoming(obj, bbelse)
            retval.add_incoming(builtin, bbif)

        else:
            retval = obj
            with cgutils.if_unlikely(self.builder, self.is_null(retval)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()

        self.incref(retval)
        return retval

    # -------------------------------------------------------------------------

    def get_builtin_obj(self, name):
        moddict = self.pyapi.get_module_dict()
        mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
        return self.builtin_lookup(mod, name)

    def builtin_lookup(self, mod, name):
        """
        Args
        ----
        mod:
            The __builtins__ dictionary or module
        name: str
            The object to lookup
        """
        fromdict = self.pyapi.dict_getitem_string(mod, name)
        self.incref(fromdict)       # fromdict is borrowed
        bbifdict = self.builder.basic_block

        with cgutils.if_unlikely(self.builder, self.is_null(fromdict)):
            # This happen if we are using the __main__ module
            frommod = self.pyapi.object_getattr_string(mod, name)

            with cgutils.if_unlikely(self.builder, self.is_null(frommod)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()

            bbifmod = self.builder.basic_block

        builtin = self.builder.phi(self.pyapi.pyobj)
        builtin.add_incoming(fromdict, bbifdict)
        builtin.add_incoming(frommod, bbifmod)

        return builtin

    def check_occurred(self):
        err_occurred = cgutils.is_not_null(self.builder,
                                           self.pyapi.err_occurred())

        with cgutils.if_unlikely(self.builder, err_occurred):
            self.return_exception_raised()

    def check_error(self, obj):
        with cgutils.if_unlikely(self.builder, self.is_null(obj)):
            self.return_exception_raised()

        return obj

    def check_int_status(self, num, ok_value=0):
        """
        Raise an exception if *num* is smaller than *ok_value*.
        """
        ok = lc.Constant.int(num.type, ok_value)
        pred = self.builder.icmp(lc.ICMP_SLT, num, ok)
        with cgutils.if_unlikely(self.builder, pred):
            self.return_exception_raised()

    def is_null(self, obj):
        return cgutils.is_null(self.builder, obj)

    def return_exception_raised(self):
        self.builder.branch(self.ehblock)

    def return_error_occurred(self):
        self.cleanup()
        self.context.return_exc(self.builder)

    def getvar(self, name, ltype=None):
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name, ltype=ltype)
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def delvar(self, name):
        """
        Delete the variable slot with the given name. This will decref
        the corresponding Python object.
        """
        ptr = self.varmap.pop(name)
        self.decref(ptr)

    def storevar(self, value, name):
        """
        Stores a llvm value and allocate stack slot if necessary.
        The llvm value can be of arbitrary type.
        """
        ptr = self.getvar(name, ltype=value.type)
        old = self.builder.load(ptr)
        assert value.type == ptr.type.pointee, (str(value.type),
                                                str(ptr.type.pointee))
        self.builder.store(value, ptr)
        # Safe to call decref even on non python object
        self.decref(old)

    def cleanup(self):
        for var in utils.dict_itervalues(self.varmap):
            self.decref(self.builder.load(var))

    def alloca(self, name, ltype=None):
        """
        Allocate a stack slot and initialize it to NULL.
        The default is to allocate a pyobject pointer.
        Use ``ltype`` to override.
        """
        if ltype is None:
            ltype = self.context.get_value_type(types.pyobject)
        bb = self.builder.basic_block
        self.builder.position_at_end(self.entry_block)
        ptr = self.builder.alloca(ltype, name=name)
        self.builder.store(cgutils.get_null_value(ltype), ptr)
        self.builder.position_at_end(bb)
        return ptr

    def incref(self, value):
        self.pyapi.incref(value)

    def decref(self, value):
        """
        This is allow to be called on non pyobject pointer, in which case
        no code is inserted.
        """
        lpyobj = self.context.get_value_type(types.pyobject)

        if value.type.kind == lc.TYPE_POINTER:
            if value.type != lpyobj:
                pass
            else:
                self.pyapi.decref(value)
