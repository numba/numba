from __future__ import print_function, division, absolute_import
import numpy
import re
from numba import types, config

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


re_typestr = re.compile(r'[<>=\|]([a-z])(\d+)?', re.I)


sizeof_unicode_char = numpy.dtype('U1').itemsize


def from_dtype(dtype):
    if dtype.fields is None:
        try:
            basetype = FROM_DTYPE[dtype]
        except KeyError:
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

            raise NotImplementedError(dtype)

        return basetype
    else:
        return from_struct_dtype(dtype)


def is_arrayscalar(val):
    return numpy.dtype(type(val)) in FROM_DTYPE


def map_arrayscalar_type(val):
    return from_dtype(numpy.dtype(type(val)))


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


# NumPy ufunc loop matching logic
# Finds out the loop that will be used (its complete type signature) when called with
# the given input types.
# ufunc - The ufunc we want to check
# op_dtypes - a string containing the dtypes of the operands using numpy char encoding.
#
# return value - the full identifier of the loop. f.e: 'dd->d' or None if no matching
#                loop is found.

def numba_types_to_numpy_letter_types(numba_type_seq):
    return [numpy.dtype(str(x)).char for x in numba_type_seq]

def numpy_letter_types_to_numba_types(numpy_letter_types_seq):
    return [from_dtype(numpy.dtype(x)) for x in numpy_letter_types_seq]

def ufunc_find_matching_loop(ufunc, op_dtypes):
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

