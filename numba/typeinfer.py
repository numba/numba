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

from __future__ import print_function
from pprint import pprint
import itertools
from numba import ir, types, utils


class TypeVar(object):
    def __init__(self, var):
        self.var = var
        self.typeset = set()
        self.locked = False

    def add_types(self, *types):
        if not types:
            return

        nbefore = len(self.typeset)
        self.typeset |= set(types)
        nafter = len(self.typeset)

        if self.locked and nafter != nbefore:
            raise Exception("Violating locked type variable")

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
        assert len(self) == 1
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
            constrain(context, typevars)


class Propagate(object):
    """
    A simple constrain for direct propagation of types for assignments.
    """

    def __init__(self, dst, src):
        self.dst = dst
        self.src = src

    def __call__(self, context, typevars):
        typevars[self.dst].union(typevars[self.src])


class CallConstrain(object):
    """Constrain for calling functions.
    Perform case analysis foreach combinations of argument types.
    """

    def __init__(self, target, func, args, kws):
        self.target = target
        self.func = func
        self.args = args
        self.kws = kws

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
            rt = context.resolve_function_type(fnty, args, ())
            assert rt is not None, "%s" % fnty
            restypes.append(rt)
        typevars[self.target].add_types(*restypes)


class IntrinsicCallConstrain(CallConstrain):
    def __call__(self, context, typevars):
        self.resolve(context, typevars, fnty=self.func)


class TypeInferer(object):
    """
    Operates on block that shares the same ir.Scope.
    """

    def __init__(self, context, blocks):
        self.context = context
        self.blocks = blocks
        self.typevars = {}
        self.constrains = ConstrainNetwork()
        # Set of assumed immutable globals
        self.assumed_immutables = set()
        # Fill the type vars
        scope = self.blocks.itervalues().next().scope
        for var in scope.localvars:
            self.typevars[var] = TypeVar(var)

    def dump(self):
        print('---- type variables ----')
        pprint(self.typevars.values())

    def seed_type(self, name, typ):
        """All arguments should be seeded.
        """
        self.typevars[name].lock(typ)

    def seed_return(self, typ):
        """Seeding of return value is optional.
        """
        for blk in self.blocks.itervalues():
            inst = blk.terminator
            if isinstance(inst, ir.Return):
                self.typevars[inst.value.name].add_types(typ)

    def build_constrain(self):
        for blk in self.blocks.itervalues():
            for inst in blk.body:
                self.constrain_statement(inst)

    def propagate(self):
        newtoken = self.get_state_token()
        oldtoken = None
        self.dump()
        # Since the number of types are finite, the typesets will eventually
        # stop growing.
        while newtoken != oldtoken:
            print("propagate".center(80, '-'))
            oldtoken = newtoken
            self.constrains.propagate(self.context, self.typevars)
            newtoken = self.get_state_token()
            self.dump()

    def unify(self):
        typdict = {}
        for var, tv in self.typevars.items():
            if len(tv) == 1:
                typdict[var] = tv.getone()
            elif len(tv) == 0:
                raise TypeError("Variable %s has no type" % var)
            else:
                typdict[var] = self.context.unify_types(*tv.get())
        return typdict

    def get_state_token(self):
        """The algorithm is monotonic.  It can only grow the typesets.
        The sum of all lengths of type sets is a cheap and accurate
        description of our progress.
        """
        return sum(len(tv) for tv in self.typevars.itervalues())

    def constrain_statement(self, inst):
        if isinstance(inst, ir.Assign):
            self.typeof_assign(inst)
        elif isinstance(inst, (ir.Jump, ir.Branch, ir.Return)):
            pass
        else:
            raise NotImplementedError(inst)

    def typeof_assign(self, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            self.typeof_const(inst, inst.target, value.value)
        elif isinstance(value, ir.Var):
            self.constrains.append(Propagate(dst=inst.target.name,
                                             src=value.name))
        elif isinstance(value, ir.Global):
            self.typeof_global(inst, inst.target, value)
        elif isinstance(value, ir.Expr):
            self.typeof_expr(inst, inst.target, value)
        else:
            raise NotImplementedError(type(value), value)

    def typeof_const(self, inst, target, const):
        if isinstance(const, int):
            if utils.bit_length(const) < 32:
                typ = types.int32
            elif utils.bit_length(const) < 64:
                typ = types.int64
            else:
                typ = types.pyobject
            self.typevars[target.name].lock(typ)
        elif isinstance(const, float):
            self.typevars[target.name].lock(types.float64)
        else:
            raise NotImplementedError(type(const))

    def typeof_global(self, inst, target, gvar):
        if gvar.name in ('range', 'xrange') and gvar.value in (range, xrange):
            self.typevars[target.name].lock(types.range_type)
            self.assumed_immutables.add(inst)
        elif gvar.value is ir.UNDEFINED:
            self.typevars[target.name].add_types(types.pyobject)
        else:
            raise NotImplementedError(gvar)

    def typeof_expr(self, inst, target, expr):
        if expr.op == 'call':
            self.typeof_call(inst, target, expr)
        elif expr.op in ('getiter', 'iternext', 'itervalid'):
            self.typeof_builtin_call(inst, target, expr.op, expr.value)
        elif expr.op == 'binop':
            self.typeof_builtin_call(inst, target, expr.fn, expr.lhs, expr.rhs)
        else:
            raise NotImplementedError(type(expr), expr)

    def typeof_call(self, inst, target, call):
        constrain = CallConstrain(target.name, call.func.name, call.args,
                                  call.kws)
        self.constrains.append(constrain)

    def typeof_builtin_call(self, inst, target, func, *args):
        constrain = IntrinsicCallConstrain(target.name, func, args, ())
        self.constrains.append(constrain)