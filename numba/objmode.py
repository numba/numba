"""
Lowering implementation for object mode.
"""

from __future__ import print_function, division, absolute_import

from llvmlite.llvmpy.core import Type, Constant
import llvmlite.llvmpy.core as lc

from numba import cgutils, ir, types, utils
from .lowering import BaseLower, ForbiddenConstruct
from .utils import builtins


# Issue #475: locals() is unsupported as calling it naively would give
# out wrong results.
_unsupported_builtins = set([locals])

# Map operators to methods on the PythonAPI class
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

        # Strings to be frozen into the Environment object
        self._frozen_strings = set()

    def pre_lower(self):
        # Store environment argument for later use
        self.envarg = self.context.get_env_argument(self.function)
        with cgutils.if_unlikely(self.builder, self.is_null(self.envarg)):
            self.pyapi.err_set_string(
                "PyExc_SystemError",
                "Numba internal error: object mode function called "
                "without an environment")
            self.return_exception_raised()
        self.env_body = self.context.get_env_body(self.builder, self.envarg)

    def post_lower(self):
        with cgutils.goto_block(self.builder, self.ehblock):
            self.cleanup()
            self.context.return_exc(self.builder)

        self._finalize_frozen_string()

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

        elif isinstance(inst, ir.SetAttr):
            target = self.loadvar(inst.target.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.object_setattr(target,
                                           self._freeze_string(inst.attr),
                                           value)
            self.check_int_status(ok)

        elif isinstance(inst, ir.DelAttr):
            target = self.loadvar(inst.target.name)
            ok = self.pyapi.object_delattr(target,
                                           self._freeze_string(inst.attr))
            self.check_int_status(ok)

        elif isinstance(inst, ir.StoreMap):
            dct = self.loadvar(inst.dct.name)
            key = self.loadvar(inst.key.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.dict_setitem(dct, key, value)
            self.check_int_status(ok)

        elif isinstance(inst, ir.Return):
            retval = self.loadvar(inst.value.name)
            # No need to incref() as the reference is already owned.
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
        if isinstance(value, (ir.Const, ir.FreeVar)):
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

    def lower_binop(self, expr, inplace=False):
        lhs = self.loadvar(expr.lhs.name)
        rhs = self.loadvar(expr.rhs.name)
        if expr.fn in PYTHON_OPMAP:
            fname = PYTHON_OPMAP[expr.fn]
            fn = getattr(self.pyapi, fname)
            res = fn(lhs, rhs, inplace=inplace)
        else:
            # Assumed to be rich comparison
            res = self.pyapi.object_richcompare(lhs, rhs, expr.fn)
        self.check_error(res)
        return res

    def lower_expr(self, expr):
        if expr.op == 'binop':
            return self.lower_binop(expr, inplace=False)
        elif expr.op == 'inplace_binop':
            return self.lower_binop(expr, inplace=True)
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
            res = self.pyapi.object_getattr(obj, self._freeze_string(expr.attr))
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
                excid = self.add_exception(ValueError)
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

        elif expr.op == 'cast':
            val = self.loadvar(expr.value.name)
            self.incref(val)
            return val

        else:
            raise NotImplementedError(expr)

    def lower_const(self, const):
        # All constants are frozen inside the environment
        index = len(self.env.consts)
        self.env.consts.append(const)
        ret = self.get_env_const(index)
        self.check_error(ret)
        self.incref(ret)
        return ret

    def lower_global(self, name, value):
        """
        1) Check global scope dictionary.
        2) Check __builtins__.
            2a) is it a dictionary (for non __main__ module)
            2b) is it a module (for __main__ module)
        """
        moddict = self.get_module_dict()
        obj = self.pyapi.dict_getitem(moddict, self._freeze_string(name))
        self.incref(obj)  # obj is borrowed

        try:
            if value in _unsupported_builtins:
                raise ForbiddenConstruct("builtins %s() is not supported"
                                         % name, loc=self.loc)
        except TypeError:
            # `value` is unhashable, ignore
            pass

        if hasattr(builtins, name):
            obj_is_null = self.is_null(obj)
            bbelse = self.builder.basic_block

            with cgutils.ifthen(self.builder, obj_is_null):
                mod = self.pyapi.dict_getitem(moddict,
                                          self._freeze_string("__builtins__"))
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

        return retval

    # -------------------------------------------------------------------------

    def get_module_dict(self):
        return self.env_body.globals

    def get_builtin_obj(self, name):
        # XXX The builtins dict could be bound into the environment
        moddict = self.get_module_dict()
        mod = self.pyapi.dict_getitem(moddict,
                                      self._freeze_string("__builtins__"))
        return self.builtin_lookup(mod, name)

    def get_env_const(self, index):
        """
        Look up constant number *index* inside the environment body.
        A borrowed reference is returned.
        """
        return self.pyapi.list_getitem(self.env_body.consts, index)

    def builtin_lookup(self, mod, name):
        """
        Args
        ----
        mod:
            The __builtins__ dictionary or module, as looked up in
            a module's globals.
        name: str
            The object to lookup
        """
        fromdict = self.pyapi.dict_getitem(mod, self._freeze_string(name))
        self.incref(fromdict)       # fromdict is borrowed
        bbifdict = self.builder.basic_block

        with cgutils.if_unlikely(self.builder, self.is_null(fromdict)):
            # This happen if we are using the __main__ module
            frommod = self.pyapi.object_getattr(mod, self._freeze_string(name))

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

    def _getvar(self, name, ltype=None):
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name, ltype=ltype)
        return self.varmap[name]

    def loadvar(self, name):
        """
        Load the llvm value of the variable named *name*.
        """
        ptr = self.varmap[name]
        val = self.builder.load(ptr)
        with cgutils.if_unlikely(self.builder, self.is_null(val)):
            self.pyapi.raise_missing_name_error(name)
            self.return_error_occurred()
        return val

    def delvar(self, name):
        """
        Delete the variable slot with the given name. This will decref
        the corresponding Python object.
        """
        ptr = self._getvar(name)  # initializes `name` if not already
        self.decref(self.builder.load(ptr))
        # This is a safety guard against double decref's, but really
        # the IR should be correct and have only one Del per variable
        # and code path.
        self.builder.store(cgutils.get_null_value(ptr.type.pointee), ptr)

    def storevar(self, value, name):
        """
        Stores a llvm value and allocate stack slot if necessary.
        The llvm value can be of arbitrary type.
        """
        ptr = self._getvar(name, ltype=value.type)
        old = self.builder.load(ptr)
        assert value.type == ptr.type.pointee, (str(value.type),
                                                str(ptr.type.pointee))
        self.builder.store(value, ptr)
        # Safe to call decref even on non python object
        self.decref(old)

    def cleanup(self):
        # Nothing to do.
        pass

    def alloca(self, name, ltype=None):
        """
        Allocate a stack slot and initialize it to NULL.
        The default is to allocate a pyobject pointer.
        Use ``ltype`` to override.
        """
        if ltype is None:
            ltype = self.context.get_value_type(types.pyobject)
        with cgutils.goto_block(self.builder, self.entry_block):
            ptr = self.builder.alloca(ltype, name=name)
            self.builder.store(cgutils.get_null_value(ltype), ptr)
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

    def _freeze_string(self, string):
        """Freeze a python string object into the code.
        Insert a reference to the Environment object later.
        """
        self._frozen_strings.add(string)
        return self.context.get_constant(types.intp, id(string)).inttoptr(
            self.pyapi.pyobj)

    def _finalize_frozen_string(self):
        """Insert all referenced string into the Environment object.
        """
        for fs in self._frozen_strings:
            self.env.consts.append(fs)
