"""
Type inference base on CPA.
The algorithm guarantees monotonic growth of type-sets for each variable.

Steps:
    1. seed initial types
    2. build constraints
    3. propagate constraints
    4. unify types

Constraint propagation is precise and does not regret (no backtracing).
Constraints push types forward following the dataflow.
"""

from __future__ import print_function, division, absolute_import

from pprint import pprint
import itertools
import traceback

from numba import ir, types, utils, config, six
from .errors import TypingError


class TypeVar(object):
    def __init__(self, context, var):
        self.context = context
        self.var = var
        self.type = None
        self.locked = False

    def add_type(self, tp):
        assert isinstance(tp, types.Type), type(tp)

        if self.locked:
            if tp != self.type:
                if self.context.can_convert(tp, self.type) is None:
                    raise TypingError("No conversion from %s to %s for "
                                      "'%s'" % (tp, self.type, self.var))
        else:
            if self.type is not None:
                unified = self.context.unify_pairs(self.type, tp)
                if unified is types.pyobject:
                    raise TypingError("cannot unify %s and %s for '%s'"
                                      % (self.type, tp, self.var))
            else:
                unified = tp
            self.type = unified

        return self.type

    def lock(self, tp):
        assert isinstance(tp, types.Type), type(tp)
        assert not self.locked

        # If there is already a type, ensure we can convert it to the
        # locked type.
        if (self.type is not None and
            self.context.can_convert(self.type, tp) is None):
            raise TypingError("No conversion from %s to %s for "
                                "'%s'" % (tp, self.type, self.var))

        self.type = tp
        self.locked = True

    def union(self, other):
        if other.type is not None:
            self.add_type(other.type)

        return self.type

    def __repr__(self):
        return '%s := %s' % (self.var, self.type)

    @property
    def defined(self):
        return self.type is not None

    def get(self):
        return (self.type,) if self.type is not None else ()

    def getone(self):
        assert self.type is not None
        return self.type

    def __len__(self):
        return 1 if self.type is not None else 0


class ConstraintNetwork(object):
    """
    TODO: It is possible to optimize constraint propagation to consider only
          dirty type variables.
    """

    def __init__(self):
        self.constraints = []

    def append(self, constraint):
        self.constraints.append(constraint)

    def propagate(self, typeinfer):
        """
        Execute all constraints.  Errors are caught and returned as a list.
        This allows progressing even though some constraints may fail
        due to lack of information (e.g. imprecise types such as List(undefined)).
        """
        errors = []
        for constraint in self.constraints:
            try:
                constraint(typeinfer)
            except TypingError as e:
                errors.append(e)
            except Exception:
                msg = "Internal error at {con}:\n{sep}\n{err}{sep}\n"
                e = TypingError(msg.format(con=constraint,
                                           err=traceback.format_exc(),
                                           sep='--%<' +'-' * 65),
                                loc=constraint.loc)
                errors.append(e)
        return errors


class Propagate(object):
    """
    A simple constraint for direct propagation of types for assignments.
    """

    def __init__(self, dst, src, loc):
        self.dst = dst
        self.src = src
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        typeinfer.copy_type(self.src, self.dst)
        # If `dst` is refined, notify us
        typeinfer.refine_map[self.dst] = self

    def refine(self, typeinfer, target_type):
        # Do not back-propagate to locked variables (e.g. constants)
        typeinfer.add_type(self.src, target_type, unless_locked=True)


class BuildTupleConstraint(object):
    def __init__(self, target, items, loc):
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        tsets = [typevars[i.name].get() for i in self.items]
        oset = typevars[self.target]
        for vals in itertools.product(*tsets):
            if vals and all(vals[0] == v for v in vals):
                tup = types.UniTuple(dtype=vals[0], count=len(vals))
            else:
                # empty tuples fall here as well
                tup = types.Tuple(vals)
            typeinfer.add_type(self.target, tup)


class BuildListConstraint(object):
    def __init__(self, target, items, loc):
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        oset = typevars[self.target]
        tsets = [typevars[i.name].get() for i in self.items]
        if not tsets:
            typeinfer.add_type(self.target, types.List(types.undefined))
        else:
            for typs in itertools.product(*tsets):
                unified = typeinfer.context.unify_types(*typs)
                typeinfer.add_type(self.target, types.List(unified))


class ExhaustIterConstraint(object):
    def __init__(self, target, count, iterator, loc):
        self.target = target
        self.count = count
        self.iterator = iterator
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        oset = typevars[self.target]
        for tp in typevars[self.iterator.name].get():
            if isinstance(tp, types.BaseTuple):
                if len(tp) == self.count:
                    typeinfer.add_type(self.target, tp)
                else:
                    raise ValueError("wrong tuple length for %r: "
                                     "expected %d, got %d"
                                     % (self.iterator.name, self.count, len(tp)))
            elif isinstance(tp, types.IterableType):
                tup = types.UniTuple(dtype=tp.iterator_type.yield_type,
                                     count=self.count)
                typeinfer.add_type(self.target, tup)


class PairFirstConstraint(object):
    def __init__(self, target, pair, loc):
        self.target = target
        self.pair = pair
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        oset = typevars[self.target]
        for tp in typevars[self.pair.name].get():
            if not isinstance(tp, types.Pair):
                # XXX is this an error?
                continue
            typeinfer.add_type(self.target, tp.first_type)


class PairSecondConstraint(object):
    def __init__(self, target, pair, loc):
        self.target = target
        self.pair = pair
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        oset = typevars[self.target]
        for tp in typevars[self.pair.name].get():
            if not isinstance(tp, types.Pair):
                # XXX is this an error?
                continue
            typeinfer.add_type(self.target, tp.second_type)


class StaticGetItemConstraint(object):
    def __init__(self, target, value, index, loc):
        self.target = target
        self.value = value
        self.index = index
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        oset = typevars[self.target]
        for tp in typevars[self.value.name].get():
            if isinstance(tp, types.BaseTuple):
                typeinfer.add_type(self.target, tp.types[self.index])


class CallConstraint(object):
    """Constraint for calling functions.
    Perform case analysis foreach combinations of argument types.
    """
    signature = None

    def __init__(self, target, func, args, kws, vararg, loc):
        self.target = target
        self.func = func
        self.args = args
        self.kws = kws or {}
        self.vararg = vararg
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        fnty = typevars[self.func].getone()
        self.resolve(typeinfer, typevars, fnty)

    def resolve(self, typeinfer, typevars, fnty):
        assert fnty
        context = typeinfer.context

        n_pos_args = len(self.args)
        kwds = [kw for (kw, var) in self.kws]
        argtypes = [typevars[a.name] for a in self.args]
        argtypes += [typevars[var.name] for (kw, var) in self.kws]
        if self.vararg is not None:
            argtypes.append(typevars[self.vararg.name])

        if not (a.defined for a in argtypes):
            # Cannot resolve call type until all argument types are known
            return

        args = tuple(a.getone() for a in argtypes)
        pos_args = args[:n_pos_args]
        if self.vararg is not None:
            if not isinstance(args[-1], types.BaseTuple):
                # Unsuitable for *args
                # (Python is more lenient and accepts all iterables)
                return
            pos_args += args[-1].types
            args = args[:-1]
        kw_args = dict(zip(kwds, args[n_pos_args:]))
        sig = context.resolve_function_type(fnty, pos_args, kw_args)
        if sig is None:
            desc = context.explain_function_type(fnty)
            headtemp = "Invalid usage of {0} with parameters ({1})"
            head = headtemp.format(fnty, ', '.join(map(str, args)))
            msg = '\n'.join([head, desc])
            raise TypingError(msg, loc=self.loc)
        typeinfer.add_type(self.target, sig.return_type)
        # If the function is a bound function and its receiver type
        # was refined, propagate it.
        if (isinstance(fnty, types.BoundFunction)
            and sig.recvr is not None
            and sig.recvr != fnty.this):
            refined_this = context.unify_pairs(sig.recvr, fnty.this)
            if refined_this.is_precise():
                refined_fnty = types.BoundFunction(fnty.template,
                                                   this=refined_this)
                typeinfer.propagate_refined_type(self.func, refined_fnty)

        self.signature = sig

    def get_call_signature(self):
        return self.signature


class IntrinsicCallConstraint(CallConstraint):
    def __call__(self, typeinfer):
        self.resolve(typeinfer, typeinfer.typevars, fnty=self.func)


class GetAttrConstraint(object):
    def __init__(self, target, attr, value, loc, inst):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc
        self.inst = inst

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        valtys = typevars[self.value.name].get()
        for ty in valtys:
            try:
                attrty = typeinfer.context.resolve_getattr(value=ty, attr=self.attr)
            except KeyError:
                args = (self.attr, ty, self.value.name, self.inst)
                msg = "Unknown attribute '%s' for %s %s %s" % args
                raise TypingError(msg, loc=self.inst.loc)
            else:
                typeinfer.add_type(self.target, attrty)
        typeinfer.refine_map[self.target] = self

    def refine(self, typeinfer, target_type):
        if isinstance(target_type, types.BoundFunction):
            recvr = target_type.this
            typeinfer.add_type(self.value.name, recvr)
            source_constraint = typeinfer.refine_map.get(self.value.name)
            if source_constraint is not None:
                source_constraint.refine(typeinfer, recvr)

    def __repr__(self):
        return 'resolving type of attribute "{attr}" of "{value}"'.format(
            value=self.value, attr=self.attr)


class SetItemConstraint(object):
    def __init__(self, target, index, value, loc):
        self.target = target
        self.index = index
        self.value = value
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        targettys = typevars[self.target.name].get()
        idxtys = typevars[self.index.name].get()
        valtys = typevars[self.value.name].get()

        for ty, it, vt in itertools.product(targettys, idxtys, valtys):
            if not typeinfer.context.resolve_setitem(target=ty,
                                                     index=it, value=vt):
                raise TypingError("Cannot resolve setitem: %s[%s] = %s" %
                                  (ty, it, vt), loc=self.loc)


class DelItemConstraint(object):
    def __init__(self, target, index, loc):
        self.target = target
        self.index = index
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        targettys = typevars[self.target.name].get()
        idxtys = typevars[self.index.name].get()

        for ty, it in itertools.product(targettys, idxtys):
            if not typeinfer.context.resolve_delitem(target=ty, index=it):
                raise TypingError("Cannot resolve delitem: %s[%s]" %
                                  (ty, it), loc=self.loc)


class SetAttrConstraint(object):
    def __init__(self, target, attr, value, loc):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc

    def __call__(self, typeinfer):
        typevars = typeinfer.typevars
        targettys = typevars[self.target.name].get()
        valtys = typevars[self.value.name].get()

        for ty, vt in itertools.product(targettys, valtys):
            if not typeinfer.context.resolve_setattr(target=ty,
                                                     attr=self.attr,
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
        self.constraints = ConstraintNetwork()

        # { index: mangled name }
        self.arg_names = {}
        self.return_type = None
        # Set of assumed immutable globals
        self.assumed_immutables = set()
        # Track all calls
        self.usercalls = []
        self.intrcalls = []
        self.delitemcalls = []
        self.setitemcalls = []
        self.setattrcalls = []
        # Target var -> constraint with refine hook
        self.refine_map = {}

        if config.DEBUG or config.DEBUG_TYPEINFER:
            self.debug = TypeInferDebug(self)
        else:
            self.debug = NullDebug()

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
        self.lock_type(name, typ)

    def seed_return(self, typ):
        """Seeding of return value is optional.
        """
        for blk in utils.itervalues(self.blocks):
            inst = blk.terminator
            if isinstance(inst, ir.Return):
                self.lock_type(inst.value.name, typ)

    def build_constraint(self):
        for blk in utils.itervalues(self.blocks):
            for inst in blk.body:
                self.constrain_statement(inst)

    def propagate(self):
        newtoken = self.get_state_token()
        oldtoken = None
        # Since the number of types are finite, the typesets will eventually
        # stop growing.
        while newtoken != oldtoken:
            self.debug.propagate_started()
            oldtoken = newtoken
            # Errors can appear when the type set is incomplete; only
            # raise them when there is no progress anymore.
            errors = self.constraints.propagate(self)
            newtoken = self.get_state_token()
            self.debug.propagate_finished()
        if errors:
            raise errors[0]

    def add_type(self, var, tp, unless_locked=False):
        assert isinstance(var, str), type(var)
        tv = self.typevars[var]
        if unless_locked and tv.locked:
            return
        unified = tv.add_type(tp)
        self.propagate_refined_type(var, unified)

    def copy_type(self, src_var, dest_var):
        unified = self.typevars[dest_var].union(self.typevars[src_var])

    def lock_type(self, var, tp):
        tv = self.typevars[var]
        tv.lock(tp)

    def propagate_refined_type(self, updated_var, updated_type):
        source_constraint = self.refine_map.get(updated_var)
        if source_constraint is not None:
            source_constraint.refine(self, updated_type)

    def unify(self):
        """
        Run the final unification pass over all inferred types, and
        catch imprecise types.
        """
        typdict = utils.UniqueDict()

        def check_var(name):
            tv = self.typevars[name]
            if not tv.defined:
                raise TypingError("Undefined variable '%s'" % (var,))
            tp = tv.getone()
            if not tp.is_precise():
                raise TypingError("Can't infer type of variable '%s': %s" % (var, tp))
            typdict[var] = tp

        # For better error display, check first user-visible vars, then
        # temporaries
        temps = set(k for k in self.typevars if not k[0].isalpha())
        others = set(self.typevars) - temps
        for var in sorted(others):
            check_var(var)
        for var in sorted(temps):
            check_var(var)

        retty = self.get_return_type(typdict)
        fntys = self.get_function_types(typdict)
        if self.generator_info:
            retty = self.get_generator_type(typdict, retty)

        self.debug.unify_finished(typdict, retty, fntys)

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
        """
        Fill and return a calltypes map using the inferred `typemap`.
        """
        # XXX why can't this be done on the fly?
        calltypes = utils.UniqueDict()
        for call, constraint in self.intrcalls:
            calltypes[call] = constraint.get_call_signature()

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

        for inst in self.delitemcalls:
            target = typemap[inst.target.name]
            index = typemap[inst.index.name]
            signature = self.context.resolve_delitem(target, index)
            calltypes[inst] = signature

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

        if rettypes:
            unified = self.context.unify_types(*rettypes)
            if not unified.is_precise():
                raise TypingError("Can't unify return type from the "
                                  "following types: %s"
                                  % ", ".join(sorted(map(str, rettypes))))
            return unified
        else:
            return types.none

    def get_state_token(self):
        """The algorithm is monotonic.  It can only grow or "refine" the
        typevar map.
        """
        return [tv.type for name, tv in sorted(self.typevars.items())]

    def constrain_statement(self, inst):
        if isinstance(inst, ir.Assign):
            self.typeof_assign(inst)
        elif isinstance(inst, ir.SetItem):
            self.typeof_setitem(inst)
        elif isinstance(inst, ir.DelItem):
            self.typeof_delitem(inst)
        elif isinstance(inst, ir.SetAttr):
            self.typeof_setattr(inst)
        elif isinstance(inst, (ir.Jump, ir.Branch, ir.Return, ir.Del)):
            pass
        elif isinstance(inst, ir.Raise):
            pass
        else:
            raise NotImplementedError(inst)

    def typeof_setitem(self, inst):
        constraint = SetItemConstraint(target=inst.target, index=inst.index,
                                       value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.setitemcalls.append(inst)

    def typeof_delitem(self, inst):
        constraint = DelItemConstraint(target=inst.target, index=inst.index,
                                       loc=inst.loc)
        self.constraints.append(constraint)
        self.delitemcalls.append(inst)

    def typeof_setattr(self, inst):
        constraint = SetAttrConstraint(target=inst.target, attr=inst.attr,
                                       value=inst.value, loc=inst.loc)
        self.constraints.append(constraint)
        self.setattrcalls.append(inst)

    def typeof_assign(self, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            self.typeof_const(inst, inst.target, value.value)
        elif isinstance(value, ir.Var):
            self.constraints.append(Propagate(dst=inst.target.name,
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
        self.constraints.append(Propagate(dst=target.name,
                                          src=src_name,
                                          loc=inst.loc))

    def typeof_const(self, inst, target, const):
        self.lock_type(target.name, self.resolve_value_type(inst, const))

    def typeof_yield(self, inst, target, yield_):
        # Sending values into generators isn't supported.
        self.add_type(target.name, types.none)

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
            self.lock_type(target.name, typ)
            self.assumed_immutables.add(inst)
        else:
            raise TypingError("Untyped global name '%s'" % gvar.name,
                              loc=inst.loc)

    def typeof_expr(self, inst, target, expr):
        if expr.op == 'call':
            if isinstance(expr.func, ir.Intrinsic):
                restype = expr.func.type.return_type
                self.add_type(target.name, restype)
                self.usercalls.append((inst.value, expr.args, expr.kws, None))
            else:
                self.typeof_call(inst, target, expr)
        elif expr.op in ('getiter', 'iternext'):
            self.typeof_intrinsic_call(inst, target, expr.op, expr.value)
        elif expr.op == 'exhaust_iter':
            constraint = ExhaustIterConstraint(target.name, count=expr.count,
                                               iterator=expr.value,
                                               loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'pair_first':
            constraint = PairFirstConstraint(target.name, pair=expr.value,
                                             loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'pair_second':
            constraint = PairSecondConstraint(target.name, pair=expr.value,
                                              loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'binop':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'inplace_binop':
            self.typeof_intrinsic_call(inst, target, expr.fn,
                                       expr.lhs, expr.rhs)
        elif expr.op == 'unary':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.value)
        elif expr.op == 'static_getitem':
            constraint = StaticGetItemConstraint(target.name, value=expr.value,
                                                 index=expr.index,
                                                 loc=expr.loc)
            self.constraints.append(constraint)
        elif expr.op == 'getitem':
            self.typeof_intrinsic_call(inst, target, expr.op, expr.value,
                                       expr.index)
        elif expr.op == 'getattr':
            constraint = GetAttrConstraint(target.name, attr=expr.attr,
                                           value=expr.value, loc=inst.loc,
                                           inst=inst)
            self.constraints.append(constraint)
        elif expr.op == 'build_tuple':
            constraint = BuildTupleConstraint(target.name, items=expr.items,
                                              loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'build_list':
            constraint = BuildListConstraint(target.name, items=expr.items,
                                             loc=inst.loc)
            self.constraints.append(constraint)
        elif expr.op == 'cast':
            self.constraints.append(Propagate(dst=target.name,
                                              src=expr.value.name,
                                              loc=inst.loc))
        else:
            raise NotImplementedError(type(expr), expr)

    def typeof_call(self, inst, target, call):
        constraint = CallConstraint(target.name, call.func.name, call.args,
                                    call.kws, call.vararg, loc=inst.loc)
        self.constraints.append(constraint)
        self.usercalls.append((inst.value, call.args, call.kws, call.vararg))

    def typeof_intrinsic_call(self, inst, target, func, *args):
        constraint = IntrinsicCallConstraint(target.name, func, args,
                                             kws=(), vararg=None, loc=inst.loc)
        self.constraints.append(constraint)
        self.intrcalls.append((inst.value, constraint))


class NullDebug(object):

    def propagate_started(self):
        pass

    def propagate_finished(self):
        pass

    def unify_finished(self, typdict, retty, fntys):
        pass


class TypeInferDebug(object):

    def __init__(self, typeinfer):
        self.typeinfer = typeinfer

    def _dump_state(self):
        print('---- type variables ----')
        pprint([v for k, v in sorted(self.typeinfer.typevars.items())])

    def propagate_started(self):
        print("propagate".center(80, '-'))

    def propagate_finished(self):
        self._dump_state()

    def unify_finished(self, typdict, retty, fntys):
        print("Variable types".center(80, "-"))
        pprint(typdict)
        print("Return type".center(80, "-"))
        pprint(retty)
        print("Call types".center(80, "-"))
        pprint(fntys)
