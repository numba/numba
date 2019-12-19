
__all__ = ['FunctionType', 'FunctionProtoType', 'numbatype']

import inspect
import types as pytypes
# TODO: implement ctypes support
# import ctypes

from .abstract import Type


class FunctionType(Type):
    """The front-end for first-class function types.
    """

    mutable = True
    cconv = None

    def __init__(self, ftype):
        if isinstance(ftype, tuple) and len(ftype) == 2:
            ftype = FunctionProtoType(ftype[0], ftype[1])
        if isinstance(ftype, pytypes.FunctionType):
            # Temporarily hold Python function until its signature can
            # be determined.
            # TODO: analyze the connection to the unboxing model,
            # why does it just works? See also get_call_type method below.
            import numba
            self.ftype = numba.njit(ftype)
            name = ftype.__name__ + '_TEMPLATE'
        elif isinstance(ftype, FunctionProtoType):
            self.ftype = ftype
            name = ftype.name
        else:
            raise TypeError(
                f'constructing {type(self).__name__} from {type(ftype)}'
                ' instance is not supported')
        super(FunctionType, self).__init__(name)

    def signature(self):
        from numba import typing
        ptype = self.ftype
        return typing.signature(ptype.rtype, *ptype.atypes)

    @property
    def key(self):
        return self.name

    def cast_python_value(self, value):
        from numba import typing
        if isinstance(value, typing.Signature):
            ptype = FunctionProtoType(value.return_type, value.args)
            return FunctionType(ptype)
        raise NotImplementedError(
            'cast_python_value({}, {})'.format(value, type(value)))

    def get_call_type(self, context, args, kws):
        if isinstance(self.ftype, FunctionProtoType):
            from numba import typing
            # TODO: match self.atypes with args
            ptype = self.ftype
            return typing.signature(ptype.rtype, *ptype.atypes)
        else:
            from numba.function import numbatype
            call_template, pysig, args, kws = self.ftype.get_call_template(
                args, kws)
            tmpl = call_template(self.ftype.typingctx)
            r = tmpl.apply(args, kws)
            self.__init__(numbatype(r).ftype)
            # self.ftype = fromobject(r).ftype
            return r
        raise NotImplementedError(self.ftype)

    def __get_call_signatures(self):
        # see explain_function_type in numba/typing/context.py
        # will be used when FunctionType is derived from Callable
        # return (), False
        raise NotImplementedError('get_call_signature()')


class FunctionProtoType(Type):
    """
    Represents a first-class function type.
    """
    mutable = True
    cconv = None

    def __init__(self, rtype, atypes):
        self.rtype = rtype
        self.atypes = tuple(atypes)

        # Note that numbatype must be able to reconstruct the function
        # type from the name.
        assert isinstance(rtype, Type), (rtype)
        lst = []
        for atype in self.atypes:
            assert isinstance(atype, Type), (atype)
            lst.append(atype.name)
        name = '%s(%s)' % (rtype, ', '.join(lst))

        super(FunctionProtoType, self).__init__(name)

    @property
    def key(self):
        return self.name


class NumbaTypeParseError(Exception):
    """Failure to parse numba type definition
    """


def numbatype(obj):
    """Return numba type from arbitrary object representing a type.

    # todo: from ctypes
    """
    import numba
    from numba import types as nbtypes

    if obj is None:
        # When in here, this usually indicates that Python function
        # signature cannot be determined.
        raise TypeError('cannot determine numba type')

    if isinstance(obj, nbtypes.Type):
        if isinstance(obj, nbtypes.Literal):
            return obj.literal_type
        return obj

    if isinstance(obj, str):
        obj = obj.strip()
        if obj in ['void', 'none', '']:
            return nbtypes.void
        if obj == 'void*':
            return nbtypes.voidptr
        t = dict(
            bool=nbtypes.boolean,
            int=nbtypes.int64,
            float=nbtypes.float32,
            complex=nbtypes.complex64,
            str=nbtypes.unicode_type,
            unicode=nbtypes.unicode_type,
        ).get(obj)
        if t is not None:
            return t
        t = getattr(nbtypes, obj, None)
        if t is not None:
            return t
        if obj.endswith('*'):
            return nbtypes.CPointer(numbatype(obj[:-1]))
        if obj.endswith(')'):
            i = _findparen(obj)
            if i < 0:
                raise NumbaTypeParseError(
                    'mismatching parenthesis in `%s`' % (obj))
            rtype = numbatype(obj[:i])
            atypes = map(numbatype, _commasplit(obj[i + 1:-1].strip()))
            ftype = FunctionProtoType(rtype, atypes)
            return FunctionType(ftype)
        if obj.startswith('{') and obj.endswith('}'):
            #return cls(*map(numbatype, _commasplit(obj[1:-1].strip())))
            pass # numba does not have a type to represent struct
        raise ValueError('Failed to construct numba type from {!r}'.format(obj))

    if isinstance(obj, numba.typing.Signature):
        rtype = numbatype(obj.return_type)
        atypes = tuple(map(numbatype, obj.args))
        ptype = FunctionProtoType(rtype, atypes)
        return FunctionType(ptype)

    if inspect.isclass(obj):
        t = {int: nbtypes.int64,
             float: nbtypes.float64,
             complex: nbtypes.complex128,
             str: nbtypes.unicode_type,
             bytes: nbtypes.Bytes}.get(obj)
        if t is not None:
            return t
        return numbatype(obj.__name__)

    if callable(obj):
        if obj.__name__ == '<lambda>':
            # lambda functions cannot carry annotations, hence:
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
        ftype = FunctionProtoType(rtype, atypes)
        return FunctionType(ftype)

    raise NotImplementedError(
        'constructing numba type from %s instance' % (type(obj)))


def _annotation_to_numba_type(annot, sig):
    if annot == sig.empty:
        return numbatype(None)
    return numbatype(annot)


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
            raise NumbaTypeParseError('failed to find lparen index in `%s`' % s)
        return i
    i = s.rfind('(', 0, j)
    if i < 0:
        raise NumbaTypeParseError('failed to find lparen index in `%s`' % s)
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
    raise NumbaTypeParseError('failed to comma-split `%s`' % s)
