from contextlib import contextmanager
from llvm.core import Constant, Type
import llvm.core as lc


true_bit = Constant.int(Type.int(1), 1)
false_bit = Constant.int(Type.int(1), 0)


class Structure(object):
    def __init__(self, context, builder, value=None):
        self._type = context.get_struct_type(self)
        self._builder = builder

        if value is None:
            self._value = alloca_once(builder, self._type)
        else:
            assert value.type.pointee == self._type, (value.type.pointee,
                                                      self._type)
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

    def __iter__(self):
        def iterator():
            for field, _ in self._fields:
                yield getattr(self, field)
        return iter(iterator())


def get_function(builder):
    return builder.basic_block.function


def get_module(builder):
    return builder.basic_block.function.module


def append_basic_block(builder, name=''):
    return get_function(builder).append_basic_block(name)


@contextmanager
def goto_block(builder, bb):
    bbold = builder.basic_block
    if bb.instructions and bb.instructions[-1].is_terminator:
        builder.position_before(bb.instructions[-1])
    else:
        builder.position_at_end(bb)
    yield
    builder.position_at_end(bbold)


@contextmanager
def goto_entry_block(builder):
    fn = get_function(builder)
    with goto_block(builder, fn.entry_basic_block):
        yield


def alloca_once(builder, ty):
    with goto_entry_block(builder):
        return builder.alloca(ty)


def terminate(builder, bbend):
    bb = builder.basic_block
    instr = bb.instructions
    if not instr or not instr[-1].is_terminator:
        builder.branch(bbend)


def get_null_value(ltype):
    return Constant.null(ltype)


def is_null(builder, val):
    null = get_null_value(val.type)
    return builder.icmp(lc.ICMP_EQ, null, val)


def is_not_null(builder, val):
    null = get_null_value(val.type)
    return builder.icmp(lc.ICMP_NE, null, val)


def set_branch_weight(builder, brinst, trueweight, falseweight):
    module = get_module(builder)
    mdid = lc.MetaDataString.get(module, "branch_weights")
    trueweight = lc.Constant.int(Type.int(), trueweight)
    falseweight = lc.Constant.int(Type.int(), falseweight)
    md = lc.MetaData.get(module, [mdid, trueweight, falseweight])
    brinst.set_metadata("prof", md)


@contextmanager
def if_unlikely(builder, pred):
    bb = builder.basic_block
    with ifthen(builder, pred):
        yield
    brinst = bb.instructions[-1]
    set_branch_weight(builder, brinst, trueweight=1, falseweight=99)


@contextmanager
def if_likely(builder, pred):
    bb = builder.basic_block
    with ifthen(builder, pred):
        yield
    brinst = bb.instructions[-1]
    set_branch_weight(builder, brinst, trueweight=99, falseweight=1)


@contextmanager
def ifthen(builder, pred):
    bb = builder.basic_block
    bbif = append_basic_block(builder, bb.name + '.if')
    bbend = append_basic_block(builder, bb.name + '.endif')
    builder.cbranch(pred, bbif, bbend)

    with goto_block(builder, bbif):
        yield bbend
        terminate(builder, bbend)

    builder.position_at_end(bbend)


def unpack_tuple(builder, tup, count):
    vals = [builder.extract_value(tup, i)
            for i in range(count)]
    return vals


def get_item_pointer(builder, aryty, ary, inds):
    # TODO only handle "any" layout for now
    strides = unpack_tuple(builder, ary.strides, count=aryty.ndim)
    dimoffs = [builder.mul(s, i) for s, i in zip(strides, inds)]
    offset = reduce(builder.add, dimoffs)
    base = builder.ptrtoint(ary.data, offset.type)
    where = builder.add(base, offset)
    ptr = builder.inttoptr(where, ary.data.type)
    return ptr
