from __future__ import absolute_import
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

        self.valmap = {}
        self.phimap = {}

    def infer(self):
        self.infer_instruction()
        self.unify_phi_nodes()
        self.check_branch()
        self.check_return_type()

    #----- internals ------

    def infer_instruction(self):
        # infer instructions
        for block in self.blocks:
            for op in block.code:
                with error_context(lineno=op.lineno,
                                   during='type inference'):
                    ty = self.op(op)
                    if ty is not None:
                        assert not hasattr(op, 'type'), "redefining type"
                        op.update(type=ty)

    def unify_phi_nodes(self):
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

    def check_branch(self):
        # check branch
        for block in self.blocks:
            term = block.terminator
            with error_context(lineno=term.lineno,
                               during='checking branch condition'):
                if term.opcode == 'branch':
                    types.boolean.coerce(term.cond.type)

    def check_return_type(self):
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
        callee = getattr(inst.callee, 'value', inst.callee)

        if isinstance(callee, Inst) and callee.type == types.method_type:
            parent = callee
            callee = '@%s' % (parent.callee[1:],)
            assert len(parent.args) == 1
            args = [parent.args[0].type] + args
            inst.update(args=list(parent.args) + list(inst.args))

        with error_context(during="resolving function %s" % callee):
            defn = self.funclib.get(callee, args)
            inst.update(defn=defn)
            if defn.return_type == types.method_type:
                inst.update(bypass=True)    # codegen should bypass this
            return defn.return_type

    def op_const(self, inst):
        ty = self.type_global(inst.value)
        if ty is None:
            raise ValueError("invalid constant value %s" % (value,))
        return ty

    def op_global(self, inst):
        value = inst.value
        ty = self.type_global(inst.value)
        if ty is None:
            return types.function_type
        else:
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

    def op_unpack(self, inst):
        source = inst.value
        etys = source.type.unpack(inst.count)
        return etys[inst.index]

    def op_tuple(self, inst):
        return types.tupletype(*(i.type for i in inst.items))

    def type_global(self, value):
        if value is None:
            return types.none_type
        elif isinstance(value, complex):
            return types.complex128
        elif isinstance(value, float):
            return types.float64
        elif isinstance(value, int):
            return types.intp
        elif isinstance(value, tuple):
            return types.tupletype(*[self.type_global(i) for i in value])

            
