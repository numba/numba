"""
Generic helpers for LLVM code generation.
"""

from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import functools
from llvm.core import Constant, Type
import llvm.core as lc
from . import errcode


true_bit = Constant.int(Type.int(1), 1)
false_bit = Constant.int(Type.int(1), 0)
true_byte = Constant.int(Type.int(8), 1)
false_byte = Constant.int(Type.int(8), 0)


def as_bool_byte(builder, value):
    return builder.zext(value, Type.int(8))


def make_anonymous_struct(builder, values):
    """
    Create an anonymous struct constant containing the given LLVM *values*.
    """
    struct_type = Type.struct([v.type for v in values])
    struct_val = Constant.undef(struct_type)
    for i, v in enumerate(values):
        struct_val = builder.insert_value(struct_val, v, i)
    return struct_val


class Structure(object):
    """
    A high-level object wrapping a alloca'ed LLVM structure, including
    named fields and attribute access.
    """

    def __init__(self, context, builder, value=None, ref=None):
        self._type = context.get_struct_type(self)
        self._context = context
        self._builder = builder
        if ref is None:
            self._value = alloca_once(builder, self._type)
            if value is not None:
                assert not is_pointer(value.type)
                assert value.type == self._type, (value.type, self._type)
                builder.store(value, self._value)
        else:
            assert value is None
            assert self._type == ref.type.pointee
            self._value = ref

        self._namemap = {}
        self._fdmap = []
        self._typemap = []
        base = Constant.int(Type.int(), 0)
        for i, (k, tp) in enumerate(self._fields):
            self._namemap[k] = i
            self._fdmap.append((base, Constant.int(Type.int(), i)))
            self._typemap.append(tp)

    def __getattr__(self, field):
        """
        Load the LLVM value of the named *field*.
        """
        if not field.startswith('_'):
            return self[self._namemap[field]]
        else:
            raise AttributeError(field)

    def __setattr__(self, field, value):
        """
        Store the LLVM *value* into the named *field*.
        """
        if field.startswith('_'):
            return super(Structure, self).__setattr__(field, value)
        self[self._namemap[field]] = value

    def __getitem__(self, index):
        """
        Load the LLVM value of the field at *index*.
        """
        offset = self._fdmap[index]
        ptr = self._builder.gep(self._value, offset)
        return self._builder.load(ptr)

    def __setitem__(self, index, value):
        """
        Store the LLVM *value* into the field at *index*.
        """
        offset = self._fdmap[index]
        ptr = self._builder.gep(self._value, offset)
        value = self._context.get_struct_member_value(self._builder,
                                                      self._typemap[index],
                                                      value)
        if ptr.type.pointee != value.type:
            raise AssertionError("Type mismatch: __setitem__(%d, ...) "
                                 "expected %r but got %r"
                                 % (index, str(ptr.type.pointee), str(value.type)))
        self._builder.store(value, ptr)

    def __len__(self):
        """
        Return the number of fields.
        """
        return len(self._namemap)

    def _getpointer(self):
        """
        Return the LLVM pointer to the underlying structure.
        """
        return self._value

    def _getvalue(self):
        """
        Load and return the value of the underlying LLVM structure.
        """
        return self._builder.load(self._value)

    # __iter__ is derived by Python from __len__ and __getitem__


def get_function(builder):
    return builder.basic_block.function


def get_module(builder):
    return builder.basic_block.function.module


def append_basic_block(builder, name=''):
    return get_function(builder).append_basic_block(name)


@contextmanager
def goto_block(builder, bb):
    bbold = builder.basic_block
    term = bb.terminator
    if term is not None:
        builder.position_before(term)
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
    if bb.terminator is None:
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
def for_range_slice(builder, start, stop, step, intp, inc=True):
    """
    Generate LLVM IR for a for-loop based on a slice  
    
    Parameters
    -------------
    builder : object
        Builder object
    start : int
        The beginning value of the slice
    stop : int
        The end value of the slice
    step : int
        The step value of the slice
    intp : 
        The data type
    inc : boolean
        A flag to handle the step < 0 case, in which case we decrement the loop

    Returns
    -----------
        None
    """

    bbcond = append_basic_block(builder, "for.cond")
    bbbody = append_basic_block(builder, "for.body")
    bbend = append_basic_block(builder, "for.end")
    bbstart = builder.basic_block
    builder.branch(bbcond)

    with goto_block(builder, bbcond):
        index = builder.phi(intp, name="loop.index")
        if (inc):
            pred = builder.icmp(lc.ICMP_SLT, index, stop)
        else:
            pred = builder.icmp(lc.ICMP_SGT, index, stop)
        builder.cbranch(pred, bbbody, bbend)

    with goto_block(builder, bbbody):
        yield index
        bbbody = builder.basic_block
        incr = builder.add(index, step)
        terminate(builder, bbcond)

    index.add_incoming(start, bbstart)
    index.add_incoming(incr, bbbody)

    builder.position_at_end(bbend)

#decrement version of for loop, start > stop, and step < 0
# as step < 0 we add rather than sub
#@contextmanager
#def for_range_slice_dec(builder, start, stop, step, intp):
#    """start """
#    bbcond = append_basic_block(builder, "for.cond")
#    bbbody = append_basic_block(builder, "for.body")
#    bbend = append_basic_block(builder, "for.end")
#
#    bbstart = builder.basic_block
#    builder.branch(bbcond)
#
#    #STEP = Constant.int(intp, 2)
#
#    with goto_block(builder, bbcond):
#        index = builder.phi(intp, name="loop.index")
#        pred = builder.icmp(lc.ICMP_SGT, index, stop)
#        builder.cbranch(pred, bbbody, bbend)
#
#    with goto_block(builder, bbbody):
#        yield index
#        bbbody = builder.basic_block
#        incr = builder.add(index, step)
#        terminate(builder, bbcond)
#
#    index.add_incoming(start, bbstart)
#    index.add_incoming(incr, bbbody)
#
#    builder.position_at_end(bbend)
#


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
    shapes = unpack_tuple(builder, ary.shape, count=aryty.ndim)
    strides = unpack_tuple(builder, ary.strides, count=aryty.ndim)
    return get_item_pointer2(builder, data=ary.data, shape=shapes,
                             strides=strides, layout=aryty.layout, inds=inds,
                             wraparound=wraparound)


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


def normalize_slice(builder, slice, length):
    """
    Clip stop
    """
    stop = slice.stop
    doclip = builder.icmp(lc.ICMP_SGT, stop, length)
    slice.stop = builder.select(doclip, length, stop)


def get_range_from_slice(builder, slicestruct):
    diff = builder.sub(slicestruct.stop, slicestruct.start)
    length = builder.sdiv(diff, slicestruct.step)
    is_neg = is_neg_int(builder, length)
    length = builder.select(is_neg, get_null_value(length.type), length)
    return length


def get_strides_from_slice(builder, ndim, strides, slice, ax):
    oldstrides = unpack_tuple(builder, strides, ndim)
    return builder.mul(slice.step, oldstrides[ax])


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
        context.return_errcode(builder, errcode.ASSERTION_ERROR)


guard_zero = guard_null


def is_struct(ltyp):
    return ltyp.kind == lc.TYPE_STRUCT


def is_pointer(ltyp):
    return ltyp.kind == lc.TYPE_POINTER


def is_struct_ptr(ltyp):
    return is_pointer(ltyp) and is_struct(ltyp.pointee)


def get_record_member(builder, record, offset, typ):
    pdata = get_record_data(builder, record)
    pval = inbound_gep(builder, pdata, 0, offset)
    assert not is_pointer(pval.type.pointee)
    return builder.bitcast(pval, Type.pointer(typ))


def get_record_data(builder, record):
    return builder.extract_value(record, 0)


def set_record_data(builder, record, buf):
    pdata = inbound_gep(builder, record, 0, 0)
    assert pdata.type.pointee == buf.type
    builder.store(buf, pdata)


def init_record_by_ptr(builder, ltyp, ptr):
    tmp = alloca_once(builder, ltyp)
    pdata = ltyp.elements[0]
    buf = builder.bitcast(ptr, pdata)
    set_record_data(builder, tmp, buf)
    return tmp


def is_neg_int(builder, val):
    return builder.icmp(lc.ICMP_SLT, val, get_null_value(val.type))


def inbound_gep(builder, ptr, *inds):
    idx = []
    for i in inds:
        if isinstance(i, int):
            ind = Constant.int(Type.int(32), i)
        else:
            ind = i
        idx.append(ind)
    return builder.gep(ptr, idx, inbounds=True)


def gep(builder, ptr, *inds):
    idx = []
    for i in inds:
        if isinstance(i, int):
            ind = Constant.int(Type.int(64), i)
        else:
            ind = i
        idx.append(ind)
    return builder.gep(ptr, idx)


# ------------------------------------------------------------------------------
# Debug

class VerboseProxy(object):
    """
    Use to wrap llvm.core.Builder to track where segfault happens
    """

    def __init__(self, obj):
        self.__obj = obj

    def __getattr__(self, key):
        fn = getattr(self.__obj, key)
        if callable(fn):
            def wrapped(*args, **kws):
                import traceback

                traceback.print_stack()
                print(key, args, kws)
                try:
                    return fn(*args, **kws)
                finally:
                    print("ok")

            return wrapped
        return fn


def printf(builder, format_string, *values):
    str_const = Constant.stringz(format_string)
    global_str_const = get_module(builder).add_global_variable(str_const.type,
                                                               '')
    global_str_const.initializer = str_const

    idx = [Constant.int(Type.int(32), 0), Constant.int(Type.int(32), 0)]
    str_addr = global_str_const.gep(idx)

    args = []
    for v in values:
        if isinstance(v, int):
            args.append(Constant.int(Type.int(), v))
        elif isinstance(v, float):
            args.append(Constant.real(Type.double(), v))

    functype = Type.function(Type.int(32), [Type.pointer(Type.int(8))], True)
    fn = get_module(builder).add_function(functype, 'printf')
    builder.call(fn, [str_addr] + args)

