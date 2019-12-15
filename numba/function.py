"""Provides numba type FunctionType that makes functions as instances
of a first-class function types.
"""
# Author: Pearu Peterson
# Created: December 2019

import types
import inspect
import numba
from numba import types as nbtypes
from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.targets.imputils import lower_constant
from numba.ccallback import CFunc
from numba import cgutils
from llvmlite import ir
# from numba.dispatcher import Dispatcher
from numba.types import FunctionType, FunctionProtoType

#
# Python objects that can be types as FunctionType are
#  - plain Python functions
#  - numba Dispatcher instances that wrap plain Python functions
#  - numba CFunc instances
#
# Disabled the first two as it interfers with existing numba tests
#


# @typeof_impl.register(types.FunctionType)
def typeof_function(val, c):
    # Python function must use annotations to specify the signature
    return fromobject(val)


# @typeof_impl.register(Dispatcher)
def typeof_Dispatcher(val, c):
    return fromobject(val.py_func)


@typeof_impl.register(CFunc)
def typeof_CFunc(val, c):
    return fromobject(val._sig)


@register_model(FunctionProtoType)
class FunctionProtoModel(models.PrimitiveModel):
    """FunctionProtoModel describes the signatures of first-class functions
    """
    def __init__(self, dmm, fe_type):
        be_type = lower_nbtype(fe_type)
        super(FunctionProtoModel, self).__init__(dmm, fe_type, be_type)


@register_model(FunctionType)
class FunctionModel(models.StructModel):
    """FunctionModel holds addresses of function implementations
    """
    def __init__(self, dmm, fe_type):
        members = [
            # address of cfunc-ded function:
            ('addr', nbtypes.intp),
            # address of PyObject* referencing the Python function
            # object, currently it is CFunc:
            ('pyaddr', nbtypes.intp),
        ]
        super(FunctionModel, self).__init__(dmm, fe_type, members)


# TODO: use numba registers? On the other hand, perhaps these are not
# needed as the FunctionModel can hold the Python functions.
pyfunc_cfunc_cache = {}
cfunc_pyfunc_cache = {}
cfunc_addr_cache = {}
addr_cfunc_cache = {}


@lower_constant(FunctionType)
def lower_constant_function_type(context, builder, typ, pyval):
    # print('LOWER_CONSTANT_function_type({}, {}, {})'.format(typ,
    # pyval, type(pyval)))
    if isinstance(pyval, CFunc):
        addr = addr_cfunc_cache.get(pyval)
        if addr is None:
            addr = pyval._wrapper_address
            cfunc_addr_cache[pyval] = addr
            addr_cfunc_cache[addr] = pyval
        sfunc = cgutils.create_struct_proxy(typ)(context, builder)
        sfunc.addr = ir.Constant(ir.IntType(64), addr)
        sfunc.pyaddr = ir.Constant(ir.IntType(64), id(pyval))
        return sfunc._getvalue()

    if isinstance(pyval, types.FunctionType):
        # TODO: make sure pyval matches with typ signature
        cfunc = pyfunc_cfunc_cache.get(pyval)
        if cfunc is None:
            cfunc = numba.cfunc(typ.signature())(pyval)
            pyfunc_cfunc_cache[pyval] = cfunc
            cfunc_pyfunc_cache[cfunc] = pyval
        return lower_constant_function_type(context, builder, typ, cfunc)

    raise NotImplementedError(
        'lower_constant_struct_function_type({}, {}, {}, {})'
        .format(context, builder, typ, pyval))


@unbox(FunctionType)
def unbox_function_type(typ, obj, c):
    # print('UNBOX_function_type({}, {}, {})'.format(typ, obj, type(obj)))
    pyaddr = c.pyapi.object_getattr_string(obj, "_wrapper_address")
    # TODO: pyaddr == NULL, e.g. when obj is pure Python function, see
    # https://github.com/numba/numba/blob/e8edec9446673a5b21a77202c0bf9a81ce5c238d/numba/targets/boxing.py#L335-L365
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    sfunc.addr = c.pyapi.long_as_long(pyaddr)
    sfunc.pyaddr = c.builder.ptrtoint(obj, lower_nbtype(nbtypes.intp))
    c.pyapi.decref(pyaddr)
    return NativeValue(sfunc._getvalue())


@box(FunctionType)
def box_function_type(typ, val, c):
    # print('BOX_function_type({}, {}, {})'.format(typ, val, type(val)))
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    pyaddr_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    c.builder.store(c.builder.inttoptr(sfunc.pyaddr, c.pyapi.pyobj), pyaddr_ptr)
    cfunc = c.builder.load(pyaddr_ptr)
    c.pyapi.incref(cfunc)
    return cfunc


#
# Utility functions, used internally here
#

def lower_nbtype(typ):
    """Return llvmlite.ir type from a numba type.
    """
    if isinstance(typ, nbtypes.Integer):
        r = ir.IntType(typ.bitwidth)
    elif isinstance(typ, FunctionType):
        r = lower_nbtype(typ.ftype)
    elif isinstance(typ, FunctionProtoType):
        rtype = lower_nbtype(typ.rtype)
        atypes = tuple(map(lower_nbtype, typ.atypes))
        r = ir.FunctionType(rtype, atypes)
    else:
        raise NotImplementedError('Lowering numba type {}'.format(typ))
    return r


class TypeParseError(Exception):
    """Failure to parse type definition
    """


def fromobject(obj):
    """Return numba type from arbitrary object representing a type.
    """
    if obj is None:
        return nbtypes.void
    if isinstance(obj, nbtypes.Type):
        return obj
    # todo: from ctypes
    if isinstance(obj, str):
        obj = obj.strip()
        if obj in ['void', 'none', '']:
            return nbtypes.void
        t = dict(
            int=nbtypes.int64,
            float=nbtypes.float64,
            complex=nbtypes.complex128).get(obj)
        if t is not None:
            return t
        t = getattr(nbtypes, obj, None)
        if t is not None:
            return t
        if obj.endswith('*'):
            return nbtypes.CPointer(fromobject(obj[:-1]))
        if obj.endswith(')'):
            i = _findparen(obj)
            if i < 0:
                raise TypeParseError('mismatching parenthesis in `%s`' % (obj))
            rtype = fromobject(obj[:i])
            atypes = map(fromobject, _commasplit(obj[i + 1:-1].strip()))
            ptype = FunctionProtoType(rtype, atypes)
            return FunctionType(ptype)
        if obj.startswith('{') and obj.endswith('}'):
            #return cls(*map(fromobject, _commasplit(obj[1:-1].strip())))
            pass # numba does not have a type to represent struct
        raise ValueError('Failed to construct numba type from {!r}'.format(obj))
    if isinstance(obj, numba.typing.Signature):
        ptype = FunctionProtoType(obj.return_type, obj.args)
        return FunctionType(ptype)
    if inspect.isclass(obj):
        t = {int: nbtypes.int64,
             float: nbtypes.float64,
             complex: nbtypes.complex128,
             str: nbtypes.unicode_type,
             bytes: nbtypes.Bytes}.get(obj)
        if t is not None:
            return t
        return fromobject(obj.__name__)
    if callable(obj):
        if obj.__name__ == '<lambda>':
            # lambda function cannot carry annotations, hence:
            raise ValueError('constructing numba type instance from '
                             'a lambda function is not supported')
        sig = inspect.signature(obj)
        rtype = _annotation_to_numba_type(sig.return_annotation, sig)
        atypes = []
        for name, param in sig.parameters.items():
            if param.kind not in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.POSITIONAL_ONLY]:
                raise ValueError(
                    'callable argument kind must be positional,'
                    ' `%s` has kind %s' % (param, param.kind))
            atype = _annotation_to_numba_type(param.annotation, sig)
            atypes.append(atype)
        ptype = FunctionProtoType(rtype, atypes)
        return FunctionType(ptype)

    raise NotImplementedError(
        'constructing numba type from %s instance' % (type(obj)))


_mangling_map = dict(
    void='v', bool='b',
    char8='c', char16='z', char32='w',
    int8='B', int16='s', int32='i', int64='l', int128='q',
    uint8='U', uint16='S', uint32='I', uint64='L', uint128='Q',
    float16='h', float32='f', float64='d', float128='x',
    complex32='H', complex64='F', complex128='D', complex256='X',
    string='t',
)


def mangle(typ):
    """Return mangled type string.

    The mangling is compatible to RBC mangling that allows unique
    demangling.
    """
    if isinstance(typ, nbtypes.NoneType):
        return 'v'
    if isinstance(typ, FunctionType):
        return '_' + mangle(typ.ftype) + 'F'
    if isinstance(typ, FunctionProtoType):
        r = mangle(typ.rtype)
        a = ''.join(mangle(a) for a in typ.atypes)
        return '_' + r + 'a' + a + 'A'
    if isinstance(typ, nbtypes.RawPointer):
        return '_vP'
    if isinstance(typ, nbtypes.CPointer):
        return '_' + mangle(typ.dtype) + 'P'
    # TODO: if typ is struct then return '_' + ''.join(map(mangle,
    # typ.members)) + 'K'
    if isinstance(typ, nbtypes.Number):
        n = _mangling_map.get(typ.name)
        if n is not None:
            return n
    raise NotImplementedError('mangle({})'.format(typ))


def _annotation_to_numba_type(annot, sig):
    if annot == sig.empty:
        return fromobject(None)
    return fromobject(annot)


def _findparen(s):
    """Find the index of left parenthesis that matches with the one at the
    end of a string.

    Used internally. Copied from rbc/typesystem.py.
    """
    j = s.find(')')
    assert j >= 0, repr((j, s))
    if j == len(s) - 1:
        i = s.find('(')
        if i < 0:
            raise TypeParseError('failed to find lparen index in `%s`' % s)
        return i
    i = s.rfind('(', 0, j)
    if i < 0:
        raise TypeParseError('failed to find lparen index in `%s`' % s)
    t = s[:i] + '_' * (j - i + 1) + s[j + 1:]
    assert len(t) == len(s), repr((t, s))
    return _findparen(t)


def _commasplit(s):
    """Split a comma-separated items taking into account parenthesis.

    Used internally. Copied from rbc/typesystem.py.
    """
    lst = s.split(',')
    ac = ''
    p1, p2 = 0, 0
    rlst = []
    for i in lst:
        p1 += i.count('(') - i.count(')')
        p2 += i.count('{') - i.count('}')
        if p1 == p2 == 0:
            rlst.append((ac + ',' + i if ac else i).strip())
            ac = ''
        else:
            ac = ac + ',' + i if ac else i
    if p1 == p2 == 0:
        return rlst
    raise TypeParseError('failed to comma-split `%s`' % s)
