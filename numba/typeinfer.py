"""
Type inference base on CPA.
The algorithm guarantees monotonic growth of type-sets for each variable.

Steps:
    1. seed initial types
    2. build constrains
    3. propagate constrains
    4. unify types

Constrain propagation is precise and does not regret (no backtracing).
Constrains push types forward following the dataflow.
"""

from __future__ import print_function, division, absolute_import

from pprint import pprint
import itertools

from numba import ir, types, utils, config, six
from numba.utils import RANGE_ITER_OBJECTS


class TypingError(Exception):
    def __init__(self, msg, loc=None):
        self.msg = msg
        self.loc = loc
        if loc:
            super(TypingError, self).__init__("%s\n%s" % (msg, loc.strformat()))
        else:
            super(TypingError, self).__init__("%s" % (msg,))


class TypeVar(object):
    def __init__(self, context, var):
        self.context = context
        self.var = var
        self.typeset = set()
        self.locked = False

    def add_types(self, *types):
        if not types:
            return

        # Sentry for None
        for ty in types:
            if ty is None:
                raise TypeError("Using None as variable type")

        nbefore = len(self.typeset)

        if self.locked:
            if set(types) != self.typeset:
                [expect] = list(self.typeset)
                for ty in types:
                    if self.context.type_compatibility(ty, expect) is None:
                        raise TypingError("No conversion from %s to %s for "
                                          "'%s'" % (ty, expect, self.var))
        else:
            self.typeset |= set(types)

        nafter = len(self.typeset)
        assert nbefore <= nafter, "Must grow monotonically"

    def lock(self, typ):
        if self.locked:
            [expect] = list(self.typeset)
            if self.context.type_compatibility(typ, expect) is None:
                raise TypingError("No conversion from %s to %s for "
                                  "'%s'" % (typ, expect, self.var))
        else:
            self.typeset = set([typ])
            self.locked = True

    def union(self, other):
        self.add_types(*other.typeset)

    def __repr__(self):
        return '%s := {%s}' % (self.var, ', '.join(map(str, self.typeset)))

    def get(self):
        return tuple(self.typeset)

    def getone(self):
        assert len(self) == 1, self.typeset
        return tuple(self.typeset)[0]

    def __len__(self):
        return len(self.typeset)


class ConstrainNetwork(object):
    """
    TODO: It is possible to optimize constrain propagation to consider only
          dirty type variables.
    """

    def __init__(self):
        self.constrains = []

    def append(self, constrain):
        self.constrains.append(constrain)

    def propagate(self, context, typevars):
        for constrain in self.constrains:
            try:
                constrain(context, typevars)
            except TypingError:
                raise
            except Exception as e:
                msg = "Internal error at {con}:\n{err}"
                raise TypingError(msg.format(con=constrain, err=e),
                                  loc=constrain.loc)


class Propagate(object):
    """
    A simple constrain for direct propagation of types for assignments.
    """

    def __init__(self, dst, src, loc):
        self.dst = dst
        self.src = src
        self.loc = loc

    def __call__(self, context, typevars):
        typevars[self.dst].union(typevars[self.src])


class BuildTupleConstrain(object):
    def __init__(self, target, items, loc):
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, context, typevars):
        tsets = [typevars[i.name].get() for i in self.items]
        oset = typevars[self.target]
        for vals in itertools.product(*tsets):
            if vals and all(vals[0] == v for v in vals):
                tup = types.UniTuple(dtype=vals[0], count=len(vals))
            else:
                # empty tuples fall here as well
                tup = types.Tuple(vals)
            oset.add_types(tup)


class ExhaustIterConstrain(object):
    def __init__(self, target, count, iterator, loc):
        self.target = target
        self.count = count
        self.iterator = iterator
        self.loc = loc

    def __call__(self, context, typevars):
        oset = typevars[self.target]
        for tp in typevars[self.iterator.name].get():
            if isinstance(tp, types.BaseTuple):
                if len(tp) == self.count:
                    oset.add_types(tp)
            elif isinstance(tp, types.IterableType):
                oset.add_types(types.UniTuple(dtype=tp.iterator_type.yield_type,
                                              count=self.count))


class PairFirstConstrain(object):
    def __init__(self, target, pair, loc):
        self.target = target
        self.pair = pair
        self.loc = loc

    def __call__(self, context, typevars):
        oset = typevars[self.target]
        for tp in typevars[self.pair.name].get():
            if not isinstance(tp, types.Pair):
                # XXX is this an error?
                continue
            oset.add_types(tp.first_type)


class PairSecondConstrain(object):
    def __init__(self, target, pair, loc):
        self.target = target
        self.pair = pair
        self.loc = loc

    def __call__(self, context, typevars):
        oset = typevars[self.target]
        for tp in typevars[self.pair.name].get():
            if not isinstance(tp, types.Pair):
                # XXX is this an error?
                continue
            oset.add_types(tp.second_type)


class StaticGetItemConstrain(object):
    def __init__(self, target, value, index, loc):
        self.target = target
        self.value = value
        self.index = index
        self.loc = loc

    def __call__(self, context, typevars):
        oset = typevars[self.target]
        for tp in typevars[self.value.name].get():
            if isinstance(tp, types.UniTuple):
                oset.add_types(tp.dtype)
            elif isinstance(tp, types.Tuple):
                oset.add_types(tp.types[self.index])


class CallConstrain(object):
    """Constrain for calling functions.
    Perform case analysis foreach combinations of argument types.
    """

    def __init__(self, target, func, args, kws, vararg, loc):
        self.target = target
        self.func = func
        self.args = args
        self.kws = kws or {}
        self.vararg = vararg
        self.loc = loc

    def __call__(self, context, typevars):
        fnty = typevars[self.func].getone()
        self.resolve(context, typevars, fnty)

    def resolve(self, context, typevars, fnty):
        assert fnty
        n_pos_args = len(self.args)
        kwds = [kw for (kw, var) in self.kws]
        argtypes = [typevars[a.name].get() for a in self.args]
        argtypes += [typevars[var.name].get() for (kw, var) in self.kws]
        if self.vararg is not None:
            argtypes.append(typevars[self.vararg.name].get())
        restypes = []
        # Case analysis for each combination of argument types.
        for args in itertools.product(*argtypes):
            pos_args = args[:n_pos_args]
            if self.vararg is not None:
                if not isinstance(args[-1], types.BaseTuple):
                    # Unsuitable for *args
                    # (Python is more lenient and accepts all iterables)
                    continue
                pos_args += args[-1].types
                args = args[:-1]
            kw_args = dict(zip(kwds, args[n_pos_args:]))
            sig = context.resolve_function_type(fnty, pos_args, kw_args)
            if sig is None:
                msg = "Undeclared %s%s" % (fnty, args)
                raise TypingError(msg, loc=self.loc)
            restypes.append(sig.return_type)
        typevars[self.target].add_types(*restypes)


class IntrinsicCallConstrain(CallConstrain):
    def __call__(self, context, typevars):
        self.resolve(context, typevars, fnty=self.func)


class GetAttrConstrain(object):
    def __init__(self, target, attr, value, loc, inst):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc
        self.inst = inst

    def __call__(self, context, typevars):
        valtys = typevars[self.value.name].get()
        restypes = []
        for ty in valtys:
            try:
                attrty = context.resolve_getattr(value=ty, attr=self.attr)
            except KeyError:
                args = (self.attr, ty, self.value.name, self.inst)
                msg = "Unknown attribute '%s' for %s %s %s" % args
                raise TypingError(msg, loc=self.inst.loc)
            else:
                restypes.append(attrty)
        typevars[self.target].add_types(*restypes)

    def __repr__(self):
        return 'resolving type of attribute "{attr}" of "{value}"'.format(
            value=self.value, attr=self.attr)


class SetItemConstrain(object):
    def __init__(self, target, index, value, loc):
        self.target = target
        self.index = index
        self.value = value
        self.loc = loc

    def __call__(self, context, typevars):
        targettys = typevars[self.target.name].get()
        idxtys = typevars[self.index.name].get()
        valtys = typevars[self.value.name].get()

        for ty, it, vt in itertools.product(targettys, idxtys, valtys):
            if not context.resolve_setitem(target=ty, index=it, value=vt):
                raise TypingError("Cannot resolve setitem: %s[%s] = %s" %
                                  (ty, it, vt), loc=self.loc)


class SetAttrConstrain(object):
    def __init__(self, target, attr, value, loc):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc

    def __call__(self, context, typevars):
        targettys = typevars[self.target.name].get()
        valtys = typevars[self.value.name].get()

        for ty, vt in itertools.product(targettys, valtys):
            if not context.resolve_setattr(target=ty, attr=self.attr,
                                           value=vt):
                raise TypingError("Cannot resolve setattr: (%s).%s = %s" %
                                  (ty, self.attr, vt), loc=self.loc)


class TypeVarMap(dict):
    def set_context(self, context):
        self.context = context

    def __getitem__(self, name):
        if name not in self:
            self[name] = TypeVar(self.context, name)
        return super(TypeVarMap, self).__getitem__(name)

    def __setitem__(self, name, value):
        assert isinstance(name, str)
        if name in self:
            raise KeyError("Cannot redefine typevar %s" % name)
        else:
            super(TypeVarMap, self).__setitem__(name, value)


class TypeInferer(object):
    """
    Operates on block that shares the same ir.Scope.
    """

    def __init__(self, context, interp):
        self.context = context
        self.blocks = interp.blocks
        self.generator_info = interp.generator_info
        self.py_func = interp.bytecode.func
        self.typevars = TypeVarMap()
        self.typevars.set_context(context)
        self.constrains = ConstrainNetwork()
        # { index: mangled name }
        self.arg_names = {}
        self.return_type = None
        # Set of assumed immutable globals
        self.assumed_immutables = set()
        # Track all calls
        self.usercalls = []
        self.intrcalls = []
        self.setitemcalls = []
        self.setattrcalls = []

    def dump(self):
        print('---- type variables ----')
        pprint(list(six.itervalues(self.typevars)))

    def _mangle_arg_name(self, name):
        # Disambiguise argument name
        return "arg.%s" % (name,)

    def seed_argument(self, name, index, typ):
        name = self._mangle_arg_name(name)
        self.seed_type(name, typ)
        self.arg_names[index] = name

    def seed_type(self, name, typ):
        """All arguments should be seeded.
        """
        self.typevars[name].lock(typ)

    def seed_return(self, typ):
        """Seeding of return value is optional.
        """
        for blk in utils.itervalues(self.blocks):
            inst = blk.terminator
            if isinstance(inst, ir.Return):
                self.typevars[inst.value.name].lock(typ)

    def build_constrain(self):
        for blk in utils.itervalues(self.blocks):
            for inst in blk.body:
                self.constrain_statement(inst)

    def propagate(self):
        newtoken = self.get_state_token()
        oldtoken = None
        if config.DEBUG:
            self.dump()
        # Since the number of types are finite, the typesets will eventually
        # stop growing.
        while newtoken != oldtoken:
            if config.DEBUG:
                print("propagate".center(80, '-'))
            oldtoken = newtoken
            self.constrains.propagate(self.context, self.typevars)
            newtoken = self.get_state_token()
            if config.DEBUG:
                self.dump()

    def unify(self):
        typdict = utils.UniqueDict()
        for var, tv in self.typevars.items():
            if len(tv) == 1:
                unified = tv.getone()
            elif len(tv) == 0:
                raise TypeError("Variable %s has no type" % var)
            else:
                unified = self.context.unify_types(*tv.get())
            if unified == types.pyobject:
                raise TypingError("Var '%s' unified to object: %s" % (var, tv))
            typdict[var] = unified
        retty = self.get_return_type(typdict)
        fntys = self.get_function_types(typdict)
        if self.generator_info:
            retty = self.get_generator_type(typdict, retty)
        return typdict, retty, fntys

    def get_generator_type(self, typdict, retty):
        gi = self.generator_info
        arg_types = [None] * len(self.arg_names)
        for index, name in self.arg_names.items():
            arg_types[index] = typdict[name]
        state_types = [typdict[var_name] for var_name in gi.state_vars]
        yield_types = [typdict[y.inst.value.name] for y in gi.get_yield_points()]
        if not yield_types:
            raise TypingError("Cannot type generator: it does not yield any value")
        yield_type = self.context.unify_types(*yield_types)
        return types.Generator(self.py_func, yield_type, arg_types, state_types,
                               has_finalizer=True)

    def get_function_types(self, typemap):
        calltypes = utils.UniqueDict()
        for call, args, kws in self.intrcalls:
            if call.op in ('inplace_binop', 'binop', 'unary'):
                fnty = call.fn
            else:
                fnty = call.op
            args = tuple(typemap[a.name] for a in args)
            assert not kws
            signature = self.context.resolve_function_type(fnty, args, ())
            assert signature is not None, (fnty, args)
            calltypes[call] = signature

        for call, args, kws, vararg in self.usercalls:
            if isinstance(call.func, ir.Intrinsic):
                signature = call.func.type
            else:
                fnty = typemap[call.func.name]

                args = tuple(typemap[a.name] for a in args)
                kws = dict((kw, typemap[var.name]) for (kw, var) in kws)
                if vararg is not None:
                    tp = typemap[vararg.name]
                    assert isinstance(tp, types.BaseTuple)
                    args = args + tp.types
                signature = self.context.resolve_function_type(fnty, args, kws)
                assert signature is not None, (fnty, args, kws, vararg)
            calltypes[call] = signature

        for inst in self.setitemcalls:
            target = typemap[inst.target.name]
            index = typemap[inst.index.name]
            value = typemap[inst.value.name]
            signature = self.context.resolve_setitem(target, index, value)
            calltypes[inst] = signature

        for inst in self.setattrcalls:
            target = typemap[inst.target.name]
            attr = inst.attr
            value = typemap[inst.value.name]
            signature = self.context.resolve_setattr(target, attr, value)
            calltypes[inst] = signature

        return calltypes

    def get_return_type(self, typemap):
        rettypes = set()
        for blk in utils.itervalues(self.blocks):
            term = blk.terminator
            if isinstance(term, ir.Return):
                rettypes.add(typemap[term.value.name])

        if types.none in rettypes:
            # Special case None return
            rettypes = rettypes - set([types.none])
            if rettypes:
                unified = self.context.unify_types(*rettypes)
                return types.Optional(unified)
            else:
                return types.none
        elif rettypes:
            unified = self.context.unify_types(*rettypes)
            return unified
        else:
            return types.none

    def get_state_token(self):
        """The algorithm is monotonic.  It can only grow the typesets.
        The sum of all lengths of type sets is a cheap and accurate
        description of our progress.
        """
        return sum(len(tv) for tv in utils.itervalues(self.typevars))

    def constrain_statement(self, inst):
        if isinstance(inst, ir.Assign):
            self.typeof_assign(inst)
        elif isinstance(inst, ir.SetItem):
            self.typeof_setitem(inst)
        elif isinstance(inst, ir.SetAttr):
            self.typeof_setattr(inst)
        elif isinstance(inst, (ir.Jump, ir.Branch, ir.Return, ir.Del)):
            pass
        elif isinstance(inst, ir.Raise):
            pass
        else:
            raise NotImplementedError(inst)

    def typeof_setitem(self, inst):
        constrain = SetItemConstrain(target=inst.target, index=inst.index,
                                     value=inst.value, loc=inst.loc)
        self.constrains.append(constrain)
        self.setitemcalls.append(inst)

    def typeof_setattr(self, inst):
        constrain = SetAttrConstrain(target=inst.target, attr=inst.attr,
                                     value=inst.value, loc=inst.loc)
        self.constrains.append(constrain)
        self.setattrcalls.append(inst)

    def typeof_assign(self, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            self.typeof_const(inst, inst.target, value.value)
        elif isinstance(value, ir.Var):
            self.constrains.append(Propagate(dst=inst.target.name,
                                             src=value.name, loc=inst.loc))
        elif isinstance(value, (ir.Global, ir.FreeVar)):
            self.typeof_global(inst, inst.target, value)
        elif isinstance(value, ir.Arg):
            self.typeof_arg(inst, inst.target, value)
        elif isinstance(value, ir.Expr):
            self.typeof_expr(inst, inst.target, value)
        elif isinstance(value, ir.Yield):
            self.typeof_yield(inst, inst.target, value)
        else:
            raise NotImplementedError(type(value), str(value))

    def resolve_value_type(self, inst, val):
        """
        Resolve the type of a simple Python value, such as can be
        represented by literals.
        """
        ty = self.context.resolve_value_type(val)
        if ty is None:
            msg = "Unsupported Python value %r" % (val,)
            raise TypingError(msg, loc=inst.loc)
        else:
            return ty

    def typeof_arg(self, inst, target, arg):
        src_name = self._mangle_arg_name(arg.name)
        self.constrains.append(Propagate(dst=target.name,
                                         src=src_name,
                                         loc=inst.loc))

    def typeof_const(self, inst, target, const):
        self.typevars[target.name].lock(self.resolve_value_type(inst, const))

    def typeof_yield(self, inst, target, yield_):
        # Sending values into generators isn't supported.
        self.typevars[target.name].add_types(types.none)

    def sentry_modified_builtin(self, inst, gvar):
        """Ensure that builtins are modified.
        """
        if (gvar.name in ('range', 'xrange') and
                    gvar.value not in utils.RANGE_ITER_OBJECTS):
            bad = True
        elif gvar.name == 'slice' and gvar.value is not slice:
            bad = True
        elif gvar.name == 'len' and gvar.value is not len:
            bad = True
        else:
            bad = False

        if bad:
            raise TypingError("Modified builtin '%s'" % gvar.name,
                              loc=inst.loc)

    def typeof_global(self, inst, target, gvar):
        typ = self.context.resolve_value_type(gvar.value)
        if isinstance(typ, types.Array):
            # Global array in nopython mode is constant
            # XXX why layout='C'?
            typ = typ.copy(layout='C', readonly=True)

        if typ is not None:
            self.sentry_modified_builtin(inst, gvar)
            self.typevars[target.name].lock(typ)
            self.assumed_immutables.add(inst)
        else:
            raise TypingError("Untyped global name '%s'" % gvar.name,
                              loc=inst.loc)

    def typeof_expr(self, inst, target, expr):
        if expr.op == 'call':
            if isinstance(expr.func, ir.Intrinsic):
                restype = expr.func.type.return_type
                self.typevars[target.name].add_types(restype)
                self.usercalls.append((inst.value, expr.args, expr.kws, None))
            else:
                self.typeof_call(inst, target, expr)
        elif expr.op in ('getiter', 'iternext'):
            self.typeof_intrinsic_call(inst, target, expr.op, expr.value)
        elif expr.op == 'exhaust_iter':
            constrain = ExhaustIterConstrain(target.name, count=expr.count,
                                             iterator=expr.value,
                                             loc=expr.loc)
            self.constrains.append(constrain)
        elif expr.op == 'pair_first':
            constrain = PairFirstConstrain(target.name, pair=expr.value,
                                           loc=expr.loc)
            self.constrains.append(constrain)
        elif expr.op == 'pair_second':
            constrain = PairSecondConstrain(target.name, pair=expr.value,
                                            loc=expr.loc)
            self.constrains.append(constrain)
        elif expr.op == 'binop':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'inplace_binop':
            # We assume type constraints for inplace operators to be the
            # same as for normal operators.  This may have to be refined in
            # the future.
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'unary':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.value)
        elif expr.op == 'static_getitem':
            constrain = StaticGetItemConstrain(target.name, value=expr.value,
                                               index=expr.index,
                                               loc=expr.loc)
            self.constrains.append(constrain)
        elif expr.op == 'getitem':
            self.typeof_intrinsic_call(inst, target, expr.op, expr.value,
                                       expr.index)
        elif expr.op == 'getattr':
            constrain = GetAttrConstrain(target.name, attr=expr.attr,
                                         value=expr.value, loc=inst.loc,
                                         inst=inst)
            self.constrains.append(constrain)
        elif expr.op == 'build_tuple':
            constrain = BuildTupleConstrain(target.name, items=expr.items,
                                            loc=inst.loc)
            self.constrains.append(constrain)
        elif expr.op == 'cast':
            self.constrains.append(Propagate(dst=target.name,
                                             src=expr.value.name,
                                             loc=inst.loc))
        else:
            raise NotImplementedError(type(expr), expr)

    def typeof_call(self, inst, target, call):
        constrain = CallConstrain(target.name, call.func.name, call.args,
                                  call.kws, call.vararg, loc=inst.loc)
        self.constrains.append(constrain)
        self.usercalls.append((inst.value, call.args, call.kws, call.vararg))

    def typeof_intrinsic_call(self, inst, target, func, *args):
        constrain = IntrinsicCallConstrain(target.name, func, args,
                                           kws=(), vararg=None, loc=inst.loc)
        self.constrains.append(constrain)
        self.intrcalls.append((inst.value, args, ()))
