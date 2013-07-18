from __future__ import absolute_import
import __builtin__
from .errors import error_context
from . import types
from .symbolic import Inst
from types import FunctionType, BuiltinFunctionType

class Infer(object):
    def __init__(self, func, blocks, funclib, args, return_type=None):
        self.func = func
        self.blocks = blocks
        self.funclib = funclib

        self.args = args
        self.return_type = return_type

        self.globals = dict(vars(__builtin__))
        self.builtins = set(self.globals.values())
        self.globals.update(self.func.func_globals)

        self.valmap = {}
        self.phimap = {}

    def infer(self):
        # infer instructions
        for block in self.blocks:
            for op in block.code:
                with error_context(lineno=op.lineno,
                                   during='type inference'):
                    ty = self.op(op)
                    if ty is not None:
                        assert not hasattr(op, 'type'), "redefining type"
                        op.update(type=ty)

        # unify phi nodes
        for op, newty in self.phimap.iteritems():
            with error_context(lineno=op.lineno,
                               during='phi unification'):
                # ensure it can be casted to phi node
                oldty = op.type
                if newty != oldty:
                    op.update(type=newty, castfrom=oldty)
                else:
                    op.update(type=newty)

        # check branch
        for block in self.blocks:
            term = block.terminator
            with error_context(lineno=term.lineno,
                               during='checking branch condition'):
                if term.opcode == 'branch':
                    types.boolean.coerce(term.cond.type)

        # check return type
        for block in self.blocks:
            term = block.terminator
            with error_context(lineno=term.lineno,
                               during='unifying return type'):
                if term.opcode == 'ret':
                    return_type = term.value.type
                    if self.return_type != return_type:
                        if (self.return_type is not None and
                                not return_type.try_coerce(self.return_type)):
                            msg = "expect return type of %s but got %s"
                            raise TypeError(msg %
                                            (self.return_type, return_type))
                        elif self.return_type is None:
                            self.return_type = return_type
                    term.update(astype=self.return_type)
                elif term.opcode == 'retvoid':
                    if self.return_type != types.void:
                        msg = "must return a value of %s"
                        raise TypeError(msg % self.return_type)

    def op(self, inst):
        attr = 'op_%s' % inst.opcode
        fn = getattr(self, attr, self.generic_op)
        return fn(inst)

    def generic_op(self, inst):
        raise NotImplementedError(inst)

    def op_arg(self, inst):
        ty = self.args[inst.name]
        return ty

    def op_load(self, inst):
        return self.valmap[inst.name]

    def op_store(self, inst):
        sty = inst.value.type
        if inst.name not in self.valmap:
            # defining
            self.valmap[inst.name] = dty = sty
        else:
            # overwriting
            dty = self.valmap[inst.name]
            sty.coerce(dty)
        inst.update(astype=dty)

    def op_call(self, inst):
        args = [v.type for v in inst.args]
        if isinstance(inst.callee, Inst):
            callee = getattr(inst.callee, 'value')
        else:
            callee = inst.callee

        if not callee:
            raise TypeError("invalid call to %s" % callee)

        with error_context(during="resolving function %s" % callee):
            defn = self.funclib.get(callee, args)
            inst.update(defn=defn)
            return defn.return_type

    def op_const(self, inst):
        if inst.value is None:
            return types.void
        elif isinstance(inst.value, complex):
            return types.complex128
        elif isinstance(inst.value, float):
            return types.float64
        elif isinstance(inst.value, int):
            return types.intp
        else:
            raise ValueError("invalid constant value %s" % inst.value)

    def op_global(self, inst):
        glbl = self.func.func_globals
        value = self.globals[inst.name]
        inst.update(value=value, type=types.function_type)
        if value not in self.builtins:
            assert False, 'XXX: inline global value: %s' % value

    def op_phi(self, inst):
        for blk, val in inst.phi:
            # use the first type definition
            ty = getattr(val, 'type')
            if ty is not None:
                break
        else:
            assert 'phi is not typed'
        for val in inst.phi.values():
            self.phimap[val] = ty
        return ty

    def op_tuple(self, inst):
        return types.tupletype(*(i.type for i in inst.items))

