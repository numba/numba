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
try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from pprint import pprint
import itertools
from numba import ir, types, utils, config, ctypes_utils, cffi_support
from numba.config import PYVERSION
from numba import numpy_support

RANGE_ITER_OBJECTS = (builtins.range,)
if PYVERSION < (3, 0):
    RANGE_ITER_OBJECTS += (builtins.xrange,)


class TypingError(Exception):
    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(TypingError, self).__init__("%s\n%s" % (msg, loc.strformat()))


class TypeVar(object):
    def __init__(self, context, var):
        self.context = context
        self.var = var
        self.typeset = set()
        self.locked = False

    def add_types(self, *types):
        if not types:
            return

        nbefore = len(self.typeset)

        if self.locked:
            if set(types) != self.typeset:
                [expect] = list(self.typeset)
                for ty in types:
                    if self.context.type_compatibility(ty, expect) is None:
                        raise TypingError("No convertsion from %s to %s" %
                                          (ty, expect))
        else:
            self.typeset |= set(types)

        nafter = len(self.typeset)
        assert nbefore <= nafter, "Must grow monotonically"

    def lock(self, typ):
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
                raise TypingError("Internal error:\n%s" % e, constrain.loc)


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
            if all(vals[0] == v for v in vals):
                tup = types.UniTuple(dtype=vals[0], count=len(vals))
            else:
                tup = types.Tuple(vals)
            oset.add_types(tup)


class CallConstrain(object):
    """Constrain for calling functions.
    Perform case analysis foreach combinations of argument types.
    """

    def __init__(self, target, func, args, kws, loc):
        self.target = target
        self.func = func
        self.args = args
        self.kws = kws
        self.loc = loc

    def __call__(self, context, typevars):
        fnty = typevars[self.func].getone()
        self.resolve(context, typevars, fnty)

    def resolve(self, context, typevars, fnty):
        assert not self.kws, "Keyword argument is not supported, yet"
        argtypes = [typevars[a.name].get() for a in self.args]
        restypes = []
        # Case analysis for each combination of argument types.
        for args in itertools.product(*argtypes):
            # TODO handling keyword arguments
            sig = context.resolve_function_type(fnty, args, ())
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
            restypes.append(attrty)
        typevars[self.target].add_types(*restypes)


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

    def __init__(self, context, blocks):
        self.context = context
        self.blocks = blocks
        self.typevars = TypeVarMap()
        self.typevars.set_context(context)
        self.constrains = ConstrainNetwork()
        self.return_type = None
        # Set of assumed immutable globals
        self.assumed_immutables = set()
        # Track all calls
        self.usercalls = []
        self.intrcalls = []
        self.setitemcalls = []

    def dump(self):
        print('---- type variables ----')
        pprint(utils.dict_values(self.typevars))

    def seed_type(self, name, typ):
        """All arguments should be seeded.
        """
        self.typevars[name].lock(typ)

    def seed_return(self, typ):
        """Seeding of return value is optional.
        """
        # self.return_type = typ
        for blk in utils.dict_itervalues(self.blocks):
            inst = blk.terminator
            if isinstance(inst, ir.Return):
                self.typevars[inst.value.name].lock(typ)
                # self.typevars[inst.value.name].lock()

    def build_constrain(self):
        for blk in utils.dict_itervalues(self.blocks):
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
                typdict[var] = tv.getone()
            elif len(tv) == 0:
                raise TypeError("Variable %s has no type" % var)
            else:
                typdict[var] = self.context.unify_types(*tv.get())

        retty = self.get_return_type(typdict)
        fntys = self.get_function_types(typdict)
        return typdict, retty, fntys

    def get_function_types(self, typemap):
        calltypes = utils.UniqueDict()
        for call, args, kws in self.intrcalls:
            if call.op in ('binop', 'unary'):
                fnty = call.fn
            else:
                fnty = call.op
            args = tuple(typemap[a.name] for a in args)
            assert not kws
            signature = self.context.resolve_function_type(fnty, args, ())
            assert signature is not None, (fnty, args)
            calltypes[call] = signature

        for call, args, kws in self.usercalls:
            fnty = typemap[call.func.name]
            args = tuple(typemap[a.name] for a in args)
            assert not kws
            signature = self.context.resolve_function_type(fnty, args, ())
            assert signature is not None, (fnty, args)
            calltypes[call] = signature

        for inst in self.setitemcalls:
            target = typemap[inst.target.name]
            index = typemap[inst.index.name]
            value = typemap[inst.value.name]
            signature = self.context.resolve_setitem(target, index, value)
            calltypes[inst] = signature

        return calltypes

    def guard_return_type(self, ty):
        if isinstance(ty, types.Array):
            msg = "Cannot return array in nopython mode"
            raise TypingError(msg, loc=self.blocks[0].loc)

    def get_return_type(self, typemap):
        rettypes = set()
        for blk in utils.dict_itervalues(self.blocks):
            term = blk.terminator
            if isinstance(term, ir.Return):
                rettypes.add(typemap[term.value.name])

        if types.none in rettypes:
            # Special case None return
            rettypes = rettypes - set([types.none])
            if rettypes:
                unified = self.context.unify_types(*rettypes)
                self.guard_return_type(unified)
                return types.Optional(unified)
            else:
                return types.none
        else:
            unified = self.context.unify_types(*rettypes)
            self.guard_return_type(unified)
            return unified

    def get_state_token(self):
        """The algorithm is monotonic.  It can only grow the typesets.
        The sum of all lengths of type sets is a cheap and accurate
        description of our progress.
        """
        return sum(len(tv) for tv in utils.dict_itervalues(self.typevars))

    def constrain_statement(self, inst):
        if isinstance(inst, ir.Assign):
            self.typeof_assign(inst)
        elif isinstance(inst, ir.SetItem):
            self.typeof_setitem(inst)
        elif isinstance(inst, (ir.Jump, ir.Branch, ir.Return, ir.Del)):
            pass
        else:
            raise NotImplementedError(inst)

    def typeof_setitem(self, inst):
        constrain = SetItemConstrain(target=inst.target, index=inst.index,
                                     value=inst.value, loc=inst.loc)
        self.constrains.append(constrain)
        self.setitemcalls.append(inst)

    def typeof_assign(self, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            self.typeof_const(inst, inst.target, value.value)
        elif isinstance(value, ir.Var):
            self.constrains.append(Propagate(dst=inst.target.name,
                                             src=value.name, loc=inst.loc))
        elif isinstance(value, ir.Global):
            self.typeof_global(inst, inst.target, value)
        elif isinstance(value, ir.Expr):
            self.typeof_expr(inst, inst.target, value)
        else:
            raise NotImplementedError(type(value), value)

    def typeof_const(self, inst, target, const):
        if const is True or const is False:
            self.typevars[target.name].lock(types.boolean)
        elif isinstance(const, (int, float)):
            ty = self.context.get_number_type(const)
            self.typevars[target.name].lock(ty)
        elif const is None:
            self.typevars[target.name].lock(types.none)
        elif isinstance(const, str):
            self.typevars[target.name].lock(types.string)
        elif isinstance(const, complex):
            self.typevars[target.name].lock(types.complex128)
        elif isinstance(const, tuple):
            tys = []
            for elem in const:
                if isinstance(elem, int):
                    tys.append(types.intp)

            if all(t == types.intp for t in tys):
                typ = types.UniTuple(types.intp, len(tys))
            else:
                typ = types.Tuple(tys)
            self.typevars[target.name].lock(typ)
        else:
            msg = "Unknown constant of type %s" % (const,)
            raise TypingError(msg, loc=inst.loc)

    def typeof_global(self, inst, target, gvar):
        if (gvar.name in ('range', 'xrange') and
                    gvar.value in RANGE_ITER_OBJECTS):
            gvty = self.context.get_global_type(gvar.value)
            self.typevars[target.name].lock(gvty)
            self.assumed_immutables.add(inst)
        if gvar.name == 'slice' and gvar.value is slice:
            gvty = self.context.get_global_type(gvar.value)
            self.typevars[target.name].lock(gvty)
            self.assumed_immutables.add(inst)
        elif gvar.name == 'len' and gvar.value is len:
            gvty = self.context.get_global_type(gvar.value)
            self.typevars[target.name].lock(gvty)
            self.assumed_immutables.add(inst)
        elif gvar.name in ('True', 'False'):
            assert gvar.value in (True, False)
            self.typevars[target.name].lock(types.boolean)
            self.assumed_immutables.add(inst)
        elif isinstance(gvar.value, (int, float)):
            gvty = self.context.get_number_type(gvar.value)
            self.typevars[target.name].lock(gvty)
            self.assumed_immutables.add(inst)
        elif numpy_support.is_arrayscalar(gvar.value):
            gvty = numpy_support.map_arrayscalar_type(gvar.value)
            self.typevars[target.name].lock(gvty)
            self.assumed_immutables.add(inst)
        elif numpy_support.is_array(gvar.value):
            ary = gvar.value
            dtype = numpy_support.from_dtype(ary.dtype)
            # force C contiguous
            gvty = types.Array(dtype, ary.ndim, 'C')
            self.typevars[target.name].lock(gvty)
            self.assumed_immutables.add(inst)
        elif ctypes_utils.is_ctypes_funcptr(gvar.value):
            cfnptr = gvar.value
            fnty = ctypes_utils.make_function_type(cfnptr)
            self.typevars[target.name].lock(fnty)
            self.assumed_immutables.add(inst)
        elif cffi_support.SUPPORTED and cffi_support.is_cffi_func(gvar.value):
            fnty = cffi_support.make_function_type(gvar.value)
            self.typevars[target.name].lock(fnty)
            self.assumed_immutables.add(inst)
        else:
            try:
                gvty = self.context.get_global_type(gvar.value)
            except KeyError:
                raise TypingError("Untyped global name '%s'" % gvar.name,
                                  loc=inst.loc)
            self.assumed_immutables.add(inst)
            self.typevars[target.name].lock(gvty)

        # TODO Hmmm...
        # elif gvar.value is ir.UNDEFINED:
        #     self.typevars[target.name].add_types(types.pyobject)

    def typeof_expr(self, inst, target, expr):
        if expr.op == 'call':
            self.typeof_call(inst, target, expr)
        elif expr.op in ('getiter', 'iternext', 'iternextsafe', 'itervalid'):
            self.typeof_intrinsic_call(inst, target, expr.op, expr.value)
        elif expr.op == 'binop':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        elif expr.op == 'unary':
            self.typeof_intrinsic_call(inst, target, expr.fn, expr.value)
        elif expr.op == 'getitem':
            self.typeof_intrinsic_call(inst, target, expr.op, expr.target,
                                       expr.index)
        elif expr.op == 'getattr':
            constrain = GetAttrConstrain(target.name, attr=expr.attr,
                                         value=expr.value, loc=inst.loc,
                                         inst=inst)
            self.constrains.append(constrain)
        elif expr.op == 'getitem':
            self.typeof_intrinsic_call(inst, target, expr.op, expr.target,
                                       expr.index, loc=inst.loc)
        elif expr.op == 'build_tuple':
            constrain = BuildTupleConstrain(target.name, items=expr.items,
                                            loc=inst.loc)
            self.constrains.append(constrain)

        else:
            raise NotImplementedError(type(expr), expr)

    def typeof_call(self, inst, target, call):
        constrain = CallConstrain(target.name, call.func.name, call.args,
                                  call.kws, loc=inst.loc)
        self.constrains.append(constrain)
        self.usercalls.append((inst.value, call.args, call.kws))

    def typeof_intrinsic_call(self, inst, target, func, *args):
        constrain = IntrinsicCallConstrain(target.name, func, args, (),
                                           loc=inst.loc)
        self.constrains.append(constrain)
        self.intrcalls.append((inst.value, args, ()))