from __future__ import print_function, division, absolute_import

import re

import numpy

from . import types, config, npdatetime
from .targets import ufunc_db

version = tuple(map(int, numpy.__version__.split('.')[:2]))
int_divbyzero_returns_zero = config.PYVERSION <= (3, 0)

FROM_DTYPE = {
    numpy.dtype('bool'): types.boolean,
    numpy.dtype('int8'): types.int8,
    numpy.dtype('int16'): types.int16,
    numpy.dtype('int32'): types.int32,
    numpy.dtype('int64'): types.int64,

    numpy.dtype('uint8'): types.uint8,
    numpy.dtype('uint16'): types.uint16,
    numpy.dtype('uint32'): types.uint32,
    numpy.dtype('uint64'): types.uint64,

    numpy.dtype('float32'): types.float32,
    numpy.dtype('float64'): types.float64,

    numpy.dtype('complex64'): types.complex64,
    numpy.dtype('complex128'): types.complex128,
}

re_typestr = re.compile(r'[<>=\|]([a-z])(\d+)?$', re.I)
re_datetimestr = re.compile(r'[<>=\|]([mM])8?(\[([a-z]+)\])?$', re.I)

sizeof_unicode_char = numpy.dtype('U1').itemsize


def _from_str_dtype(dtype):
    m = re_typestr.match(dtype.str)
    if not m:
        raise NotImplementedError(dtype)
    groups = m.groups()
    typecode = groups[0]
    if typecode == 'U':
        # unicode
        if dtype.byteorder not in '=|':
            raise NotImplementedError("Does not support non-native "
                                      "byteorder")
        count = dtype.itemsize // sizeof_unicode_char
        assert count == int(groups[1]), "Unicode char size mismatch"
        return types.UnicodeCharSeq(count)

    elif typecode == 'S':
        # char
        count = dtype.itemsize
        assert count == int(groups[1]), "Char size mismatch"
        return types.CharSeq(count)

    else:
        raise NotImplementedError(dtype)


def _from_datetime_dtype(dtype):
    m = re_datetimestr.match(dtype.str)
    if not m:
        raise NotImplementedError(dtype)
    groups = m.groups()
    typecode = groups[0]
    unit = groups[2] or ''
    if typecode == 'm':
        return types.NPTimedelta(unit)
    elif typecode == 'M':
        return types.NPDatetime(unit)
    else:
        raise NotImplementedError(dtype)


def from_dtype(dtype):
    """
    Return a Numba Type instance corresponding to the given Numpy *dtype*.
    NotImplementedError is raised on unsupported Numpy dtypes.
    """
    if dtype.fields is None:
        try:
            return FROM_DTYPE[dtype]
        except KeyError:
            if dtype.char in 'SU':
                return _from_str_dtype(dtype)
            if dtype.char in 'mM' and npdatetime.NPDATETIME_SUPPORTED:
                return _from_datetime_dtype(dtype)
            raise NotImplementedError(dtype)
    else:
        return from_struct_dtype(dtype)


def is_arrayscalar(val):
    return numpy.dtype(type(val)) in FROM_DTYPE


def map_arrayscalar_type(val):
    if isinstance(val, numpy.generic):
        # We can't blindly call numpy.dtype() as it loses information
        # on some types, e.g. datetime64 and timedelta64.
        dtype = val.dtype
    else:
        try:
            dtype = numpy.dtype(type(val))
        except TypeError:
            raise NotImplementedError("no corresponding numpy dtype for %r" % type(val))
    return from_dtype(dtype)


def is_array(val):
    return isinstance(val, numpy.ndarray)


def map_layout(val):
    if val.flags['C_CONTIGUOUS']:
        layout = 'C'
    elif val.flags['F_CONTIGUOUS']:
        layout = 'F'
    else:
        layout = 'A'
    return layout


def supported_ufunc_loop(ufunc, loop):
    """returns whether the loop for the ufunc is supported -in nopython-

    For ufuncs implemented using the ufunc_db, it is supported if the ufunc_dbc
    contains a lowering definition for 'loop' in the 'ufunc' entry.

    For other ufuncs, it is type based. The loop will be considered valid if it
    only contains the following letter types: '?bBhHiIlLqQfd'. Note this is
    legacy and when implementing new ufuncs the ufunc_db should be preferred,
    as it allows for a more fine-grained incremental support.
    """
    try:
        # check if the loop has a codegen description in the
        # ufunc_db. If so, we can proceed.

        # note that as of now not all ufuncs have an entry in the
        # ufunc_db
        supported_loop = loop in ufunc_db.ufunc_db[ufunc]
    except KeyError:
        # for ufuncs not in ufunc_db, base the decision of whether the
        # loop is supported on its types
        loop_types = loop[:ufunc.nin] + loop[-ufunc.nout:]
        supported_types = '?bBhHiIlLqQfd'
        # check if all the types involved in the ufunc loop are
        # supported in this mode
        supported_loop =  all((t in supported_types for t in loop_types))

    return supported_loop


def numba_types_to_numpy_letter_types(numba_type_seq):
    return [numpy.dtype(str(x)).char for x in numba_type_seq]


def numpy_letter_types_to_numba_types(numpy_letter_types_seq):
    return [from_dtype(numpy.dtype(x)) for x in numpy_letter_types_seq]


def ufunc_find_matching_loop(ufunc, op_dtypes):
    """Find the appropriate loop to be used for a ufunc based on the types
    of the operands

    ufunc        - The ufunc we want to check
    op_dtypes    - a string containing the dtypes of the operands using
                   numpy char encoding.
    return value - the full identifier of the loop. f.e: 'dd->d' or
                   None if no matching loop is found.
    """
    assert(isinstance(ufunc, numpy.ufunc))
    assert(len(op_dtypes) == ufunc.nin)

    # In NumPy, the loops are evaluated from first to last. The first one that is viable
    # is the one used. One loop is viable if it is possible to cast every operand to the
    # one expected by the ufunc. Note that the output is not considered in this logic.
    for candidate in ufunc.types:
        if numpy.alltrue([numpy.can_cast(*x) for x in zip(op_dtypes, candidate[0:ufunc.nin])]):
            # found
            return candidate

    return None


def from_struct_dtype(dtype):
    if dtype.hasobject:
        raise TypeError("Do not support object containing dtype")

    fields = {}
    for name, (elemdtype, offset) in dtype.fields.items():
        fields[name] = from_dtype(elemdtype), offset

    size = dtype.itemsize
    align = dtype.alignment

    return types.Record(str(dtype.descr), fields, size, align, dtype)
