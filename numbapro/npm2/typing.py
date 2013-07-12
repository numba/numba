from .errors import error_context

class Type(object):
    def __new__(cls, obj):
        if isinstance(obj, Type):
            return obj
        else:
            cls.check_protocol(obj)
            instance = object.__new__(cls)
            instance.obj = obj
            return instance

    @classmethod
    def check_protocol(cls, obj):
        assert hasattr(obj, 'coerce'), 'coerce method must be defined'

    def coerce(self, other, noraise=False):
        ret = self.obj.coerce(Type(other).obj)
        if ret is None and not noraise:
            raise TypeError('can not coerce %s -> %s' % (other, self))
        return ret


class Infer(object):
    def __init__(self, symbolic, intp, args, return_type=None):
        self.symbolic = symbolic
        self.args = args
        self.return_type = return_type

        self.valmap = {}
        self.typemap = {}

    def infer(self):
        for block in self.symbolic.blocks:
            self.curblock = block
            for op in block.code:
                with error_context(lineno=op.lineno):
                    self.typemap[op] = self.op(op)
        return self.typemap


    def op(self, inst):
        attr = 'op_%s' % inst.opcode
        fn = getattr(self, attr, self.generic_op)
        fn(inst)

    def generic_op(self, inst):
        raise NotImplementedError(inst)

    def op_arg(self, inst):
        ty = self.args[inst.name]
        self.assign(inst.name, ty)
        return ty

    def op_load(self, inst):
        self.valmap[inst.name]
