from __future__ import absolute_import
import __builtin__
from .errors import error_context
from . import types
from .symbolic import Inst
from types import FunctionType, BuiltinFunctionType

class Infer(object):
    def __init__(self, symbolic, funclib, args, return_type=None):
        self.symbolic = symbolic
        self.funclib = funclib

        self.args = args
        self.return_type = return_type

        self.globals = dict(vars(__builtin__))
        self.globals.update(self.symbolic.func.func_globals)

        self.valmap = {}
        self.phimap = {}

    def infer(self):
        for block in self.symbolic.blocks:
            self.curblock = block
            for op in block.code:
                with error_context(lineno=op.lineno,
                                   when='type inference'):
                    ty = self.op(op)
                    if ty is not None:
                        assert not hasattr(op, 'type'), "redefining type"
                        op.update(type=ty)


        for op, newty in self.phimap.iteritems():
            with error_context(lineno=op.lineno,
                               when='phi unification'):
                # ensure it can be casted to phi node
                oldty = op.type
                if newty != oldty:
                    op.update(type=newty, castfrom=oldty)
                else:
                    op.update(type=newty)

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
            self.valmap[inst.name] = sty
        else:
            # overwriting
            dty = self.valmap[inst.name]
            sty.coerce(dty)

    def op_call(self, inst):
        args = [v.type for v in inst.args]
        if isinstance(inst.callee, Inst):
            callee = getattr(inst.callee, 'value')
        else:
            callee = inst.callee

        if not callee:
            raise TypeError("invalid call to %s" % callee)

        with error_context(when="resolving function %s" % callee):
            defn = self.funclib.get(callee, args)
            inst.update(defn=defn)
            return defn.return_type

    def op_const(self, inst):
        if isinstance(inst.value, complex):
            return types.complex128
        elif isinstance(inst.value, float):
            return types.float64
        elif isinstance(inst.value, int):
            return types.intp
        else:
            raise ValueError("invalid constant value %s" % inst.value)

    def op_global(self, inst):
        glbl = self.symbolic.func.func_globals
        value = self.globals[inst.name]
        inst.update(value=value)
        if not isinstance(value, (FunctionType, BuiltinFunctionType)):
            assert False, 'XXX: inline global value'

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
