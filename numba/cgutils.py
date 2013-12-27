from contextlib import contextmanager
from llvm.core import Constant, Type


class Structure(object):
    def __init__(self, context, builder, value=None):
        self._type = context.get_struct_type(self)
        self._builder = builder

        if value is None:
            with goto_entry_block(builder):
                    self._value = builder.alloca(self._type)
        else:
            assert value.type.pointee == self._type
            self._value = value

        self._fdmap = {}
        base = Constant.int(Type.int(), 0)
        for i, (k, _) in enumerate(self._fields):
            self._fdmap[k] = (base, Constant.int(Type.int(), i))

    def __getattr__(self, field):
        offset = self._fdmap[field]
        ptr = self._builder.gep(self._value, offset)
        return self._builder.load(ptr)

    def __setattr__(self, field, value):
        if field.startswith('_'):
            return super(Structure, self).__setattr__(field, value)
        offset = self._fdmap[field]
        ptr = self._builder.gep(self._value, offset)
        assert ptr.type.pointee == value.type
        self._builder.store(value, ptr)

    def _getvalue(self):
        return self._value


def get_function(builder):
    return builder.basic_block.function


def append_basic_block(builder, name=''):
    return get_function(builder).append_basic_block(name)


@contextmanager
def goto_block(builder, bb):
    bbold = builder.basic_block
    builder.position_at_end(bb)
    yield
    builder.position_at_end(bbold)


@contextmanager
def goto_entry_block(builder):
    fn = get_function(builder)
    with goto_block(builder, fn.entry_basic_block):
        yield


def terminate(builder, bbend):
    bb = builder.basic_block
    instr = bb.instructions
    if not instr or not instr[-1].is_terminator:
        builder.branch(bbend)


@contextmanager
def ifthen(builder, pred):
    bbif = append_basic_block(builder, 'if')
    bbend = append_basic_block(builder, 'endif')
    builder.cbranch(pred, bbif, bbend)

    with goto_block(builder, bbif):
        yield bbend
        terminate(builder, bbend)

    builder.position_at_end(bbend)


