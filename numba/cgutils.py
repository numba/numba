from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import functools
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
        if not field.startswith('_'):
            offset = self._fdmap[field]
            ptr = self._builder.gep(self._value, offset)
            return self._builder.load(ptr)
        else:
            raise AttributeError(field)

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


def alloca_once(builder, ty, name=''):
    with goto_entry_block(builder):
        return builder.alloca(ty, name=name)


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


is_true = is_not_null
is_false = is_null


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


@contextmanager
def ifnot(builder, pred):
    with ifthen(builder, builder.not_(pred)):
        yield


@contextmanager
def ifelse(builder, pred, expect=None):
    bbtrue = append_basic_block(builder, 'if.true')
    bbfalse = append_basic_block(builder, 'if.false')
    bbendif = append_basic_block(builder, 'endif')

    br = builder.cbranch(pred, bbtrue, bbfalse)
    if expect is not None:
        if expect:
            set_branch_weight(builder, br, trueweight=99, falseweight=1)
        else:
            set_branch_weight(builder, br, trueweight=1, falseweight=99)

    then = IfBranchObj(builder, bbtrue, bbendif)
    otherwise = IfBranchObj(builder, bbfalse, bbendif)

    yield then, otherwise

    builder.position_at_end(bbendif)


class IfBranchObj(object):
    def __init__(self, builder, bbenter, bbend):
        self.builder = builder
        self.bbenter = bbenter
        self.bbend = bbend

    def __enter__(self):
        self.builder.position_at_end(self.bbenter)

    def __exit__(self, exc_type, exc_val, exc_tb):
        terminate(self.builder, self.bbend)


@contextmanager
def for_range(builder, count, intp):
    start = Constant.int(intp, 0)
    stop = count

    bbcond = append_basic_block(builder, "for.cond")
    bbbody = append_basic_block(builder, "for.body")
    bbend = append_basic_block(builder, "for.end")

    bbstart = builder.basic_block
    builder.branch(bbcond)

    ONE = Constant.int(intp, 1)

    with goto_block(builder, bbcond):
        index = builder.phi(intp, name="loop.index")
        pred = builder.icmp(lc.ICMP_SLT, index, stop)
        builder.cbranch(pred, bbbody, bbend)

    with goto_block(builder, bbbody):
        yield index
        bbbody = builder.basic_block
        incr = builder.add(index, ONE)
        terminate(builder, bbcond)

    index.add_incoming(start, bbstart)
    index.add_incoming(incr, bbbody)

    builder.position_at_end(bbend)


@contextmanager
def loop_nest(builder, shape, intp):
    with _loop_nest(builder, shape, intp) as indices:
        assert len(indices) == len(shape)
        yield indices


@contextmanager
def _loop_nest(builder, shape, intp):
    with for_range(builder, shape[0], intp) as ind:
        if len(shape) > 1:
            with _loop_nest(builder, shape[1:], intp) as indices:
                yield (ind,) + indices
        else:
            yield (ind,)


def pack_array(builder, values):
    n = len(values)
    ty = values[0].type
    ary = Constant.undef(Type.array(ty, n))
    for i, v in enumerate(values):
        ary = builder.insert_value(ary, v, i)
    return ary


def unpack_tuple(builder, tup, count):
    vals = [builder.extract_value(tup, i)
            for i in range(count)]
    return vals


def get_item_pointer(builder, aryty, ary, inds, wraparound=False):
    if wraparound:
        # Wraparound
        shapes = unpack_tuple(builder, ary.shape, count=aryty.ndim)
        indices = []
        for ind, dimlen in zip(inds, shapes):
            ZERO = Constant.null(ind.type)
            negative = builder.icmp(lc.ICMP_SLT, ind, ZERO)
            wrapped = builder.add(dimlen, ind)
            selected = builder.select(negative, wrapped, ind)
            indices.append(selected)
    else:
        indices = inds
    del inds
    intp = indices[0].type
    # Indexing code
    if aryty.layout == 'C':
        # C contiguous
        shapes = unpack_tuple(builder, ary.shape, count=aryty.ndim)
        steps = []
        for i in range(len(shapes)):
            last = Constant.int(intp, 1)
            for j in shapes[i + 1:]:
                last = builder.mul(last, j)
            steps.append(last)

        loc = Constant.int(intp, 0)
        for i, s in zip(indices, steps):
            tmp = builder.mul(i, s)
            loc = builder.add(loc, tmp)
        ptr = builder.gep(ary.data, [loc])
        return ptr
    else:
        # Any layout
        strides = unpack_tuple(builder, ary.strides, count=aryty.ndim)
        dimoffs = [builder.mul(s, i) for s, i in zip(strides, indices)]
        offset = functools.reduce(builder.add, dimoffs)
        base = builder.ptrtoint(ary.data, offset.type)
        where = builder.add(base, offset)
        ptr = builder.inttoptr(where, ary.data.type)
        return ptr


def get_item_pointer2(builder, data, shape, strides, layout, inds,
                      wraparound=False):
    if wraparound:
        # Wraparound
        indices = []
        for ind, dimlen in zip(inds, shape):
            ZERO = Constant.null(ind.type)
            negative = builder.icmp(lc.ICMP_SLT, ind, ZERO)
            wrapped = builder.add(dimlen, ind)
            selected = builder.select(negative, wrapped, ind)
            indices.append(selected)
    else:
        indices = inds
    del inds
    intp = indices[0].type
    # Indexing code
    if layout in 'CF':
        steps = []
        # Compute steps for each dimension
        if layout == 'C':
            # C contiguous
            for i in range(len(shape)):
                last = Constant.int(intp, 1)
                for j in shape[i + 1:]:
                    last = builder.mul(last, j)
                steps.append(last)
        elif layout == 'F':
            # F contiguous
            for i in range(len(shape)):
                last = Constant.int(intp, 1)
                for j in shape[:i]:
                    last = builder.mul(last, j)
                steps.append(last)
        else:
            raise Exception("unreachable")
        # Compute index
        loc = Constant.int(intp, 0)
        for i, s in zip(indices, steps):
            tmp = builder.mul(i, s)
            loc = builder.add(loc, tmp)
        ptr = builder.gep(data, [loc])
        return ptr
    else:
        # Any layout
        dimoffs = [builder.mul(s, i) for s, i in zip(strides, indices)]
        offset = functools.reduce(builder.add, dimoffs)
        base = builder.ptrtoint(data, offset.type)
        where = builder.add(base, offset)
        ptr = builder.inttoptr(where, data.type)
        return ptr


class MetadataKeyStore(object):
    def __init__(self, module, name):
        self.module = module
        self.key = name
        self.nmd = self.module.get_or_insert_named_metadata("python.module")

    def set(self, value):
        """
        Add a string value
        """
        md = lc.MetaData.get(self.module,
                             [lc.MetaDataString.get(self.module, value)])
        self.nmd.add(md)

    def get(self):
        """
        Get string value
        """
        node = self.nmd._ptr.getOperand(0)
        return lc._make_value(node.getOperand(0)).string


def is_scalar_zero(builder, value):
    nullval = Constant.null(value.type)
    if value.type in (Type.float(), Type.double()):
        isnull = builder.fcmp(lc.FCMP_OEQ, nullval, value)
    else:
        isnull = builder.icmp(lc.ICMP_EQ, nullval, value)
    return isnull


def guard_null(context, builder, value):
    with if_unlikely(builder, is_scalar_zero(builder, value)):
        context.return_errcode(builder, 1)


guard_zero = guard_null