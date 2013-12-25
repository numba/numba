from __future__ import print_function
from pprint import pprint
from numba import ir, types, utils


class TypeVar(object):
    def __init__(self, var):
        self.var = var
        self.typeset = set()
        self.locked = False

    def add_types(self, *types):
        if self.locked:
            if set(types) != self.typeset:
                raise Exception("Violating locked typevar")
        self.typeset |= set(types)

    def lock(self, typ):
        self.typeset = set([typ])
        self.locked = True

    def __repr__(self):
        return '%s := {%s}' % (self.var, ', '.join(map(str, self.typeset)))


class ConstrainNetwork(object):
    def __init__(self, context, typevars):
        self.context = context
        self.typevars = typevars
        self.constrains = []

    def append(self, constrain):
        self.constrains.append(constrain)


class Propagate(object):
    def __init__(self, dst, src):
        self.dst = dst
        self.src = src

    def __call__(self, context, typevars):
        raise NotImplementedError


class CallConstrain(object):
    def __init__(self, target, func, args, kws):
        self.target = target
        self.func = func
        self.args = args
        self.kws = kws

    def __call__(self, context, typevars):
        raise NotImplementedError


class TypeInferer(object):
    def __init__(self, context, blocks):
        self.context = context
        self.blocks = blocks
        self.typevars = {}
        self.constrains = ConstrainNetwork(context=context,
                                           typevars=self.typevars)
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
        self.typevars[name].lock(typ)

    def seed_return(self, typ):
        for blk in self.blocks.itervalues():
            inst = blk.body.terminator
            if isinstance(inst, ir.Return):
                self.seed_type(inst.value.name, typ)


    def build_constrain(self):
        for blk in self.blocks.itervalues():
            for inst in blk.body:
                self.constrain_statement(inst)

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
            self.typeof_global(inst, inst.target, value.value)
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

    def typeof_global(self, inst, target, value):
        if value in ('range', 'xrange'):
            self.typevars[target.name].lock(types.range_type)
            self.assumed_immutables.add(inst)

    def typeof_expr(self, inst, target, expr):
        if expr.op == 'call':
            self.typeof_call(inst, target, expr)
        elif expr.op in ('getiter', 'iternext', 'itervalid'):
            self.typeof_builtin_call(inst, target, expr.op, expr.value)
        elif expr.op == 'binop':
            self.typeof_builtin_call(inst, target, expr.fn, (expr.lhs,
                                                             expr.rhs))
        else:
            raise NotImplementedError(type(expr), expr)

    def typeof_call(self, inst, target, call):
        constrain = CallConstrain(target.name, call.func, call.args, call.kws)
        self.constrains.append(constrain)

    def typeof_builtin_call(self, inst, target, func, args):
        constrain = CallConstrain(target.name, func, args, ())
        self.constrains.append(constrain)