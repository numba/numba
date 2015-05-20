"""
Generic helpers for LLVM code generation.
"""

from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import functools
import re

from llvmlite import ir
from llvmlite.llvmpy.core import Constant, Type
import llvmlite.llvmpy.core as lc

from . import utils


true_bit = Constant.int(Type.int(1), 1)
false_bit = Constant.int(Type.int(1), 0)
true_byte = Constant.int(Type.int(8), 1)
false_byte = Constant.int(Type.int(8), 0)


def as_bool_byte(builder, value):
    return builder.zext(value, Type.int(8))


def as_bool_bit(builder, value):
    return builder.icmp(lc.ICMP_NE, value, Constant.null(value.type))


def make_anonymous_struct(builder, values, struct_type=None):
    """
    Create an anonymous struct containing the given LLVM *values*.
    """
    if struct_type is None:
        struct_type = Type.struct([v.type for v in values])
    struct_val = Constant.undef(struct_type)
    for i, v in enumerate(values):
        struct_val = builder.insert_value(struct_val, v, i)
    return struct_val


def make_bytearray(buf):
    """
    Make a byte array constant from *buf*.
    """
    b = bytearray(buf)
    n = len(b)
    return ir.Constant(ir.ArrayType(ir.IntType(8), n), b)

_struct_proxy_cache = {}


def create_struct_proxy(fe_type):
    """
    Returns a specialized StructProxy subclass for the given fe_type.
    """
    res = _struct_proxy_cache.get(fe_type)
    if res is None:
        clsname = StructProxy.__name__ + '_' + str(fe_type)
        bases = (StructProxy,)
        clsmembers = dict(_fe_type=fe_type)
        res = type(clsname, bases, clsmembers)
        _struct_proxy_cache[fe_type] = res
    return res


class StructProxy(object):
    """
    Creates a `Structure` like interface that is constructed with information
    from DataModel instance.  FE type must have a data model that is a
    subclass of StructModel.
    """
    # The following class members must be overridden by subclass
    _fe_type = None

    def __init__(self, context, builder, value=None, ref=None):
        from numba import datamodel   # Avoid circular import
        self._context = context
        self._dmm = self._context.data_model_manager
        self._datamodel = self._dmm[self._fe_type]
        if not isinstance(self._datamodel, datamodel.StructModel):
            raise TypeError("Not a structure model: {0}".format(self._datamodel))
        self._builder = builder

        self._be_type = self._datamodel.get_value_type()
        assert not is_pointer(self._be_type)

        if ref is not None:
            assert value is None
            assert ref.type.pointee == self._be_type
            self._value = ref
        else:
            self._value = alloca_once(self._builder, self._be_type, zfill=True)
            if value is not None:
                self._builder.store(value, self._value)

    def _get_ptr_by_index(self, index):
        geped = self._builder.gep(self._value,
                                  [Constant.int(Type.int(), 0),
                                   Constant.int(Type.int(), index)])
        return geped

    def _get_ptr_by_name(self, attrname):
        index = self._datamodel.get_field_position(attrname)
        return self._get_ptr_by_index(index)

    def __getattr__(self, field):
        """
        Load the LLVM value of the named *field*.
        """
        if not field.startswith('_'):
            return self[self._datamodel.get_field_position(field)]
        else:
            raise AttributeError(field)

    def __setattr__(self, field, value):
        """
        Store the LLVM *value* into the named *field*.
        """
        if field.startswith('_'):
            return super(StructProxy, self).__setattr__(field, value)
        self[self._datamodel.get_field_position(field)] = value

    def __getitem__(self, index):
        """
        Load the LLVM value of the field at *index*.
        """

        return self._builder.load(self._get_ptr_by_index(index))

    def __setitem__(self, index, value):
        """
        Store the LLVM *value* into the field at *index*.
        """
        ptr = self._get_ptr_by_index(index)
        self._builder.store(value, ptr)

    def __len__(self):
        """
        Return the number of fields.
        """
        return self._datamodel.field_count

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

    def _setvalue(self, value):
        """Store the value in this structure"""
        assert not is_pointer(value.type)
        assert value.type == self._type, (value.type, self._type)
        self._builder.store(value, self._value)


class Structure(object):
    """
    A high-level object wrapping a alloca'ed LLVM structure, including
    named fields and attribute access.
    """

    # XXX Should this warrant several separate constructors?
    def __init__(self, context, builder, value=None, ref=None, cast_ref=False):
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
            assert is_pointer(ref.type)
            if self._type != ref.type.pointee:
                if cast_ref:
                    ref = builder.bitcast(ref, Type.pointer(self._type))
                else:
                    raise TypeError(
                        "mismatching pointer type: got %s, expected %s"
                        % (ref.type.pointee, self._type))
            self._value = ref

        self._namemap = {}
        self._fdmap = []
        self._typemap = []
        base = Constant.int(Type.int(), 0)
        for i, (k, tp) in enumerate(self._fields):
            self._namemap[k] = i
            self._fdmap.append((base, Constant.int(Type.int(), i)))
            self._typemap.append(tp)

    def _get_ptr_by_index(self, index):
        ptr = self._builder.gep(self._value, self._fdmap[index])
        return ptr

    def _get_ptr_by_name(self, attrname):
        return self._get_ptr_by_index(self._namemap[attrname])

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

        return self._builder.load(self._get_ptr_by_index(index))

    def __setitem__(self, index, value):
        """
        Store the LLVM *value* into the field at *index*.
        """
        ptr = self._get_ptr_by_index(index)
        if ptr.type.pointee != value.type:
            fmt = "Type mismatch: __setitem__(%d, ...) expected %r but got %r"
            raise AssertionError(fmt % (index,
                                        str(ptr.type.pointee),
                                        str(value.type)))
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

    def _setvalue(self, value):
        """Store the value in this structure"""
        assert not is_pointer(value.type)
        assert value.type == self._type, (value.type, self._type)
        self._builder.store(value, self._value)

    # __iter__ is derived by Python from __len__ and __getitem__


def get_function(builder):
    return builder.basic_block.function


def get_module(builder):
    return builder.basic_block.function.module


def append_basic_block(builder, name=''):
    return get_function(builder).append_basic_block(name)


@contextmanager
def goto_block(builder, bb):
    """
    A context manager which temporarily positions *builder* at the end
    of basic block *bb* (but before any terminator).
    """
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


def alloca_once(builder, ty, size=None, name='', zfill=False):
    """Allocate stack memory at the entry block of the current function
    pointed by ``builder`` withe llvm type ``ty``.  The optional ``size`` arg
    set the number of element to allocate.  The default is 1.  The optional
    ``name`` arg set the symbol name inside the llvm IR for debugging.
    If ``zfill`` is set, also filling zeros to the memory.
    """
    with goto_entry_block(builder):
        ptr = builder.alloca(ty, size=size, name=name)
        if zfill:
            builder.store(Constant.null(ty), ptr)
        return ptr


def alloca_once_value(builder, value, name=''):
    """
    Like alloca_once(), but passing a *value* instead of a type.  The
    type is inferred and the allocated slot is also initialized with the
    given value.
    """
    storage = alloca_once(builder, value.type)
    builder.store(value, storage)
    return storage


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
    bbif = append_basic_block(builder, add_postfix(bb.name, '.if'))
    bbend = append_basic_block(builder, add_postfix(bb.name, '.endif'))
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
    inc : boolean, optional
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
    if not indices:
        # Indexing with empty tuple
        return builder.gep(data, [get_null_value(Type.int(32))])
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
        return pointer_add(builder, data, offset)


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


def is_scalar_zero(builder, value):
    """
    Return a predicate representing whether *value* is equal to zero.
    """
    assert not is_pointer(value.type)
    assert not is_struct(value.type)
    nullval = Constant.null(value.type)
    if value.type in (Type.float(), Type.double()):
        isnull = builder.fcmp(lc.FCMP_OEQ, nullval, value)
    else:
        isnull = builder.icmp(lc.ICMP_EQ, nullval, value)
    return isnull


def is_not_scalar_zero(builder, value):
    """
    Return a predicate representin whether a *value* is not equal to zero.
    not exactly "not is_scalar_zero" because of nans
    """
    assert not is_pointer(value.type)
    assert not is_struct(value.type)
    nullval = Constant.null(value.type)
    if value.type in (Type.float(), Type.double()):
        isnull = builder.fcmp(lc.FCMP_UNE, nullval, value)
    else:
        isnull = builder.icmp(lc.ICMP_NE, nullval, value)
    return isnull


def is_scalar_zero_or_nan(builder, value):
    """
    Return a predicate representing whether *value* is equal to either zero
    or NaN.
    """
    assert not is_pointer(value.type)
    assert not is_struct(value.type)
    nullval = Constant.null(value.type)
    if value.type in (Type.float(), Type.double()):
        isnull = builder.fcmp(lc.FCMP_UEQ, nullval, value)
    else:
        isnull = builder.icmp(lc.ICMP_EQ, nullval, value)
    return isnull

is_true = is_not_scalar_zero
is_false = is_scalar_zero

def is_scalar_neg(builder, value):
    """is _value_ negative?. Assumes _value_ is signed"""
    nullval = Constant.null(value.type)
    if value.type in (Type.float(), Type.double()):
        isneg = builder.fcmp(lc.FCMP_OLT, value, nullval)
    else:
        isneg = builder.icmp(lc.ICMP_SLT, value, nullval)
    return isneg


def guard_null(context, builder, value, exc_tuple):
    """
    Guard against *value* being null or zero.
    *exc_tuple* should be a (exception type, arguments...) tuple.
    """
    with if_unlikely(builder, is_scalar_zero(builder, value)):
        exc = exc_tuple[0]
        exc_args = exc_tuple[1:] or None
        context.call_conv.return_user_exc(builder, exc, exc_args)


guard_zero = guard_null


def is_struct(ltyp):
    """
    Whether the LLVM type *typ* is a pointer type.
    """
    return ltyp.kind == lc.TYPE_STRUCT


def is_pointer(ltyp):
    """
    Whether the LLVM type *typ* is a struct type.
    """
    return ltyp.kind == lc.TYPE_POINTER


def is_struct_ptr(ltyp):
    """
    Whether the LLVM type *typ* is a pointer-to-struct type.
    """
    return is_pointer(ltyp) and is_struct(ltyp.pointee)


def get_record_member(builder, record, offset, typ):
    pval = inbound_gep(builder, record, 0, offset)
    assert not is_pointer(pval.type.pointee)
    return builder.bitcast(pval, Type.pointer(typ))


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


def gep(builder, ptr, *inds, **kws):
    name = kws.pop('name', '')
    idx = []
    for i in inds:
        if isinstance(i, int):
            # NOTE: llvm only accepts int32 inside structs, not int64
            ind = Constant.int(Type.int(32), i)
        else:
            ind = i
        idx.append(ind)
    return builder.gep(ptr, idx, name=name)


def pointer_add(builder, ptr, offset, return_type=None):
    """
    Add an integral *offset* to pointer *ptr*, and return a pointer
    of *return_type* (or, if omitted, the same type as *ptr*).

    Note the computation is done in bytes, and ignores the width of
    the pointed item type.
    """
    intptr_t = Type.int(utils.MACHINE_BITS)
    intptr = builder.ptrtoint(ptr, intptr_t)
    if isinstance(offset, int):
        offset = Constant.int(intptr_t, offset)
    intptr = builder.add(intptr, offset)
    return builder.inttoptr(intptr, return_type or ptr.type)


def memset(builder, ptr, size, value):
    """
    Fill *size* bytes starting from *ptr* with *value*.
    """
    sizety = size.type
    memset = "llvm.memset.p0i8.i%d" % (sizety.width)
    module = get_module(builder)
    i32 = lc.Type.int(32)
    i8 = lc.Type.int(8)
    i1 = lc.Type.int(1)
    # void @llvm.memset.p0i8.iXY(i8* <dest>, i8 <val>,
    #                            iXY <len>, i32 <align>, i1 <isvolatile>)
    ptr = builder.bitcast(ptr, lc.Type.pointer(i8))
    fnty = lc.Type.function(lc.Type.void(),
                            [ptr.type, i8, sizety, i32, i1])
    fn = module.get_or_insert_function(fnty, name=memset)
    if isinstance(value, int):
        value = Constant.int(i8, value)
    builder.call(fn, [ptr, value, size,
                      Constant.int(i32, 0), Constant.int(i1, 0)])


def global_constant(builder_or_module, name, value, linkage=lc.LINKAGE_INTERNAL):
    """
    Get or create a (LLVM module-)global constant with *name* or *value*.
    """
    if isinstance(builder_or_module, lc.Module):
        module = builder_or_module
    else:
        module = get_module(builder_or_module)
    data = module.add_global_variable(value.type, name=name)
    data.linkage = linkage
    data.global_constant = True
    data.initializer = value
    return data


def divmod_by_constant(builder, val, divisor):
    """
    Compute the (quotient, remainder) of *val* divided by the constant
    positive *divisor*.  The semantics reflects those of Python integer
    floor division, rather than C's / LLVM's signed division and modulo.
    The difference lies with a negative *val*.
    """
    assert divisor > 0
    divisor = Constant.int(val.type, divisor)
    one = Constant.int(val.type, 1)

    quot = alloca_once(builder, val.type)

    with ifelse(builder, is_neg_int(builder, val)) as (if_neg, if_pos):
        with if_pos:
            # quot = val / divisor
            quot_val = builder.sdiv(val, divisor)
            builder.store(quot_val, quot)
        with if_neg:
            # quot = -1 + (val + 1) / divisor
            val_plus_one = builder.add(val, one)
            quot_val = builder.sdiv(val_plus_one, divisor)
            builder.store(builder.sub(quot_val, one), quot)

    # rem = val - quot * divisor
    # (should be slightly faster than a separate modulo operation)
    quot_val = builder.load(quot)
    rem_val = builder.sub(val, builder.mul(quot_val, divisor))
    return quot_val, rem_val


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
        else:
            args.append(v)
    functype = Type.function(Type.int(32), [Type.pointer(Type.int(8))], True)
    fn = get_module(builder).get_or_insert_function(functype, 'printf')
    builder.call(fn, [str_addr] + args)


def cbranch_or_continue(builder, cond, bbtrue):
    """
    Branch conditionally or continue.

    Note: a new block is created and builder is moved to the end of the new
          block.
    """
    fn = get_function(builder)
    bbcont = fn.append_basic_block('.continue')
    builder.cbranch(cond, bbtrue, bbcont)
    builder.position_at_end(bbcont)
    return bbcont


def add_postfix(name, postfix):
    """Add postfix to string.  If the postfix is already there, add a counter.
    """
    regex = "(.*{0})([0-9]*)$".format(postfix)
    m = re.match(regex, name)
    if m:
        head, ct = m.group(1), m.group(2)
        if len(ct):
            ct = int(ct) + 1
        else:
            ct = 1

        return "{head}{ct}".format(head=head, ct=ct)
    return name + postfix


def memcpy(builder, dst, src, count):
    """
    Emit a memcpy to the builder.

    Copies each element of dst to src. Unlike the C equivalent, each element
    can be any LLVM type.

    Assumes
    -------
    * dst.type == src.type
    * count is positive
    """
    assert dst.type == src.type
    with for_range(builder, count, count.type) as idx:
        out_ptr = builder.gep(dst, [idx])
        in_ptr = builder.gep(src, [idx])
        builder.store(builder.load(in_ptr), out_ptr)
