from __future__ import print_function, division, absolute_import

import collections
import ctypes
import re

import numpy as np

from . import errors, types, config, npdatetime, utils


version = tuple(map(int, np.__version__.split('.')[:2]))
int_divbyzero_returns_zero = config.PYVERSION <= (3, 0)

# Starting from Numpy 1.10, ufuncs accept argument conversion according
# to the "same_kind" rule (used to be "unsafe").
strict_ufunc_typing = version >= (1, 10)


FROM_DTYPE = {
    np.dtype('bool'): types.boolean,
    np.dtype('int8'): types.int8,
    np.dtype('int16'): types.int16,
    np.dtype('int32'): types.int32,
    np.dtype('int64'): types.int64,

    np.dtype('uint8'): types.uint8,
    np.dtype('uint16'): types.uint16,
    np.dtype('uint32'): types.uint32,
    np.dtype('uint64'): types.uint64,

    np.dtype('float32'): types.float32,
    np.dtype('float64'): types.float64,

    np.dtype('complex64'): types.complex64,
    np.dtype('complex128'): types.complex128,
}

re_typestr = re.compile(r'[<>=\|]([a-z])(\d+)?$', re.I)
re_datetimestr = re.compile(r'[<>=\|]([mM])8?(\[([a-z]+)\])?$', re.I)

sizeof_unicode_char = np.dtype('U1').itemsize


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
    if type(dtype) == type and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype)
    elif getattr(dtype, "fields", None) is not None:
        return from_struct_dtype(dtype)

    try:
        return FROM_DTYPE[dtype]
    except KeyError:
        char = dtype.char

        if char in 'SU':
            return _from_str_dtype(dtype)
        if char in 'mM':
            return _from_datetime_dtype(dtype)
        if char in 'V':
            subtype = from_dtype(dtype.subdtype[0])
            return types.NestedArray(subtype, dtype.shape)

    raise NotImplementedError(dtype)


_as_dtype_letters = {
    types.NPDatetime: 'M8',
    types.NPTimedelta: 'm8',
    types.CharSeq: 'S',
    types.UnicodeCharSeq: 'U',
}


def as_dtype(nbtype):
    """
    Return a numpy dtype instance corresponding to the given Numba type.
    NotImplementedError is if no correspondence is known.
    """
    nbtype = types.unliteral(nbtype)
    if isinstance(nbtype, (types.Complex, types.Integer, types.Float)):
        return np.dtype(str(nbtype))
    if nbtype is types.bool_:
        return np.dtype('?')
    if isinstance(nbtype, (types.NPDatetime, types.NPTimedelta)):
        letter = _as_dtype_letters[type(nbtype)]
        if nbtype.unit:
            return np.dtype('%s[%s]' % (letter, nbtype.unit))
        else:
            return np.dtype(letter)
    if isinstance(nbtype, (types.CharSeq, types.UnicodeCharSeq)):
        letter = _as_dtype_letters[type(nbtype)]
        return np.dtype('%s%d' % (letter, nbtype.count))
    if isinstance(nbtype, types.Record):
        return as_struct_dtype(nbtype)
    if isinstance(nbtype, types.EnumMember):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.npytypes.DType):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.NumberClass):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.NestedArray):
        spec = (as_dtype(nbtype.dtype), tuple(nbtype.shape))
        return np.dtype(spec)
    raise NotImplementedError("%r cannot be represented as a Numpy dtype"
                              % (nbtype,))

def as_struct_dtype(rec):
    """Convert Numba Record type to NumPy structured dtype
    """
    assert isinstance(rec, types.Record)
    names = [k for k, _ in rec.members]
    fields = {
        'names': names,
        'formats': [as_dtype(t) for _, t in rec.members],
        'offsets': [rec.offset(k) for k in names],
        'itemsize': rec.size,
    }
    return np.dtype(fields, align=rec.aligned)

def is_arrayscalar(val):
    return np.dtype(type(val)) in FROM_DTYPE


def map_arrayscalar_type(val):
    if isinstance(val, np.generic):
        # We can't blindly call np.dtype() as it loses information
        # on some types, e.g. datetime64 and timedelta64.
        dtype = val.dtype
    else:
        try:
            dtype = np.dtype(type(val))
        except TypeError:
            raise NotImplementedError("no corresponding numpy dtype for %r" % type(val))
    return from_dtype(dtype)


def is_array(val):
    return isinstance(val, np.ndarray)


def map_layout(val):
    if val.flags['C_CONTIGUOUS']:
        layout = 'C'
    elif val.flags['F_CONTIGUOUS']:
        layout = 'F'
    else:
        layout = 'A'
    return layout


def select_array_wrapper(inputs):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    select the appropriate input for wrapping the operation output,
    according to each input's __array_priority__.

    An index into *inputs* is returned.
    """
    max_prio = float('-inf')
    selected_input = None
    selected_index = None
    for index, ty in enumerate(inputs):
        # Ties are broken by choosing the first winner, as in Numpy
        if isinstance(ty, types.ArrayCompatible) and ty.array_priority > max_prio:
            selected_input = ty
            selected_index = index
            max_prio = ty.array_priority

    assert selected_index is not None
    return selected_index


def resolve_output_type(context, inputs, formal_output):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    and the operation's formal output type (a types.Array instance),
    resolve the actual output type using the typing *context*.

    This uses a mechanism compatible with Numpy's __array_priority__ /
    __array_wrap__.
    """
    selected_input = inputs[select_array_wrapper(inputs)]
    args = selected_input, formal_output
    sig = context.resolve_function_type('__array_wrap__', args, {})
    if sig is None:
        if selected_input.array_priority == types.Array.array_priority:
            # If it's the same priority as a regular array, assume we
            # should return the output unchanged.
            # (we can't define __array_wrap__ explicitly for types.Buffer,
            #  as that would be inherited by most array-compatible objects)
            return formal_output
        raise errors.TypingError("__array_wrap__ failed for %s" % (args,))
    return sig.return_type


def supported_ufunc_loop(ufunc, loop):
    """Return whether the *loop* for the *ufunc* is supported -in nopython-.

    *loop* should be a UFuncLoopSpec instance, and *ufunc* a numpy ufunc.

    For ufuncs implemented using the ufunc_db, it is supported if the ufunc_db
    contains a lowering definition for 'loop' in the 'ufunc' entry.

    For other ufuncs, it is type based. The loop will be considered valid if it
    only contains the following letter types: '?bBhHiIlLqQfd'. Note this is
    legacy and when implementing new ufuncs the ufunc_db should be preferred,
    as it allows for a more fine-grained incremental support.
    """
    from .targets import ufunc_db
    loop_sig = loop.ufunc_sig
    try:
        # check if the loop has a codegen description in the
        # ufunc_db. If so, we can proceed.

        # note that as of now not all ufuncs have an entry in the
        # ufunc_db
        supported_loop = loop_sig in ufunc_db.get_ufunc_info(ufunc)
    except KeyError:
        # for ufuncs not in ufunc_db, base the decision of whether the
        # loop is supported on its types
        loop_types = [x.char for x in loop.numpy_inputs + loop.numpy_outputs]
        supported_types = '?bBhHiIlLqQfd'
        # check if all the types involved in the ufunc loop are
        # supported in this mode
        supported_loop =  all(t in supported_types for t in loop_types)

    return supported_loop


class UFuncLoopSpec(collections.namedtuple('_UFuncLoopSpec',
                                           ('inputs', 'outputs', 'ufunc_sig'))):
    """
    An object describing a ufunc loop's inner types.  Properties:
    - inputs: the inputs' Numba types
    - outputs: the outputs' Numba types
    - ufunc_sig: the string representing the ufunc's type signature, in
      Numpy format (e.g. "ii->i")
    """

    __slots__ = ()

    @property
    def numpy_inputs(self):
        return [as_dtype(x) for x in self.inputs]

    @property
    def numpy_outputs(self):
        return [as_dtype(x) for x in self.outputs]


def ufunc_can_cast(from_, to, has_mixed_inputs, casting='safe'):
    """
    A variant of np.can_cast() that can allow casting any integer to
    any real or complex type, in case the operation has mixed-kind
    inputs.

    For example we want `np.power(float32, int32)` to be computed using
    SP arithmetic and return `float32`.
    However, `np.sqrt(int32)` should use DP arithmetic and return `float64`.
    """
    from_ = np.dtype(from_)
    to = np.dtype(to)
    if has_mixed_inputs and from_.kind in 'iu' and to.kind in 'cf':
        # Decide that all integers can cast to any real or complex type.
        return True
    return np.can_cast(from_, to, casting)


def ufunc_find_matching_loop(ufunc, arg_types):
    """Find the appropriate loop to be used for a ufunc based on the types
    of the operands

    ufunc        - The ufunc we want to check
    arg_types    - The tuple of arguments to the ufunc, including any
                   explicit output(s).
    return value - A UFuncLoopSpec identifying the loop, or None
                   if no matching loop is found.
    """

    # Separate logical input from explicit output arguments
    input_types = arg_types[:ufunc.nin]
    output_types = arg_types[ufunc.nin:]
    assert(len(input_types) == ufunc.nin)

    try:
        np_input_types = [as_dtype(x) for x in input_types]
    except NotImplementedError:
        return None
    try:
        np_output_types = [as_dtype(x) for x in output_types]
    except NotImplementedError:
        return None

    # Whether the inputs are mixed integer / floating-point
    has_mixed_inputs = (
        any(dt.kind in 'iu' for dt in np_input_types) and
        any(dt.kind in 'cf' for dt in np_input_types))

    def choose_types(numba_types, ufunc_letters):
        """
        Return a list of Numba types representing *ufunc_letters*,
        except when the letter designates a datetime64 or timedelta64,
        in which case the type is taken from *numba_types*.
        """
        assert len(ufunc_letters) >= len(numba_types)
        types = [tp if letter in 'mM' else from_dtype(np.dtype(letter))
                 for tp, letter in zip(numba_types, ufunc_letters)]
        # Add missing types (presumably implicit outputs)
        types += [from_dtype(np.dtype(letter))
                  for letter in ufunc_letters[len(numba_types):]]
        return types

    # In NumPy, the loops are evaluated from first to last. The first one
    # that is viable is the one used. One loop is viable if it is possible
    # to cast every input operand to the one expected by the ufunc.
    # Also under NumPy 1.10+ the output must be able to be cast back
    # to a close enough type ("same_kind").

    for candidate in ufunc.types:
        ufunc_inputs = candidate[:ufunc.nin]
        ufunc_outputs = candidate[-ufunc.nout:]
        if 'O' in ufunc_inputs:
            # Skip object arrays
            continue
        found = True
        # Skip if any input or output argument is mismatching
        for outer, inner in zip(np_input_types, ufunc_inputs):
            # (outer is a dtype instance, inner is a type char)
            if outer.char in 'mM' or inner in 'mM':
                # For datetime64 and timedelta64, we want to retain
                # precise typing (i.e. the units); therefore we look for
                # an exact match.
                if outer.char != inner:
                    found = False
                    break
            elif not ufunc_can_cast(outer.char, inner,
                                    has_mixed_inputs, 'safe'):
                found = False
                break
        if found and strict_ufunc_typing:
            # Can we cast the inner result to the outer result type?
            for outer, inner in zip(np_output_types, ufunc_outputs):
                if (outer.char not in 'mM' and not
                    ufunc_can_cast(inner, outer.char,
                                   has_mixed_inputs, 'same_kind')):
                    found = False
                    break
        if found:
            # Found: determine the Numba types for the loop's inputs and
            # outputs.
            try:
                inputs = choose_types(input_types, ufunc_inputs)
                outputs = choose_types(output_types, ufunc_outputs)
            except NotImplementedError:
                # One of the selected dtypes isn't supported by Numba
                # (e.g. float16), try other candidates
                continue
            else:
                return UFuncLoopSpec(inputs, outputs, candidate)

    return None


def _is_aligned_struct(struct):
    return struct.isalignedstruct


def from_struct_dtype(dtype):
    """Convert a NumPy structured dtype to Numba Record type
    """
    if dtype.hasobject:
        raise TypeError("Do not support dtype containing object")

    fields = []
    for name, info in dtype.fields.items():
        # *info* may have 3 element if it has a "title", which can be ignored
        [elemdtype, offset] = info[:2]
        ty = from_dtype(elemdtype)
        infos = {
            'type': ty,
            'offset': offset,
        }
        fields.append((name, infos))

    # Note: dtype.alignment is not consistent.
    #       It is different after passing into a recarray.
    #       recarray(N, dtype=mydtype).dtype.alignment != mydtype.alignment
    size = dtype.itemsize
    aligned = _is_aligned_struct(dtype)

    return types.Record(fields, size, aligned)


def _get_bytes_buffer(ptr, nbytes):
    """
    Get a ctypes array of *nbytes* starting at *ptr*.
    """
    if isinstance(ptr, ctypes.c_void_p):
        ptr = ptr.value
    arrty = ctypes.c_byte * nbytes
    return arrty.from_address(ptr)

def _get_array_from_ptr(ptr, nbytes, dtype):
    return np.frombuffer(_get_bytes_buffer(ptr, nbytes), dtype)


def carray(ptr, shape, dtype=None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in C order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
    from .typing.ctypes_utils import from_ctypes

    try:
        # Use ctypes parameter protocol if available
        ptr = ptr._as_parameter_
    except AttributeError:
        pass

    # Normalize dtype, to accept e.g. "int64" or np.int64
    if dtype is not None:
        dtype = np.dtype(dtype)

    if isinstance(ptr, ctypes.c_void_p):
        if dtype is None:
            raise TypeError("explicit dtype required for void* argument")
        p = ptr
    elif isinstance(ptr, ctypes._Pointer):
        ptrty = from_ctypes(ptr.__class__)
        assert isinstance(ptrty, types.CPointer)
        ptr_dtype = as_dtype(ptrty.dtype)
        if dtype is not None and dtype != ptr_dtype:
            raise TypeError("mismatching dtype '%s' for pointer %s"
                            % (dtype, ptr))
        dtype = ptr_dtype
        p = ctypes.cast(ptr, ctypes.c_void_p)
    else:
        raise TypeError("expected a ctypes pointer, got %r" % (ptr,))

    nbytes = dtype.itemsize * np.product(shape, dtype=np.intp)
    return _get_array_from_ptr(p, nbytes, dtype).reshape(shape)


def farray(ptr, shape, dtype=None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in Fortran order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
    if not isinstance(shape, utils.INT_TYPES):
        shape = shape[::-1]
    return carray(ptr, shape, dtype).T


def is_contiguous(dims, strides, itemsize):
    """Is the given shape, strides, and itemsize of C layout?

    Note: The code is usable as a numba-compiled function
    """
    nd = len(dims)
    # Check and skip 1s or 0s in inner dims
    innerax = nd - 1
    while innerax > -1 and dims[innerax] <= 1:
        innerax -= 1

    # Early exit if all axis are 1s or 0s
    if innerax < 0:
        return True

    # Check itemsize matches innermost stride
    if itemsize != strides[innerax]:
        return False

    # Check and skip 1s or 0s in outer dims
    outerax = 0
    while outerax < innerax and dims[outerax] <= 1:
        outerax += 1

    # Check remaining strides to be contiguous
    ax = innerax
    while ax > outerax:
        if strides[ax] * dims[ax] != strides[ax - 1]:
            return False
        ax -= 1
    return True


def is_fortran(dims, strides, itemsize):
    """Is the given shape, strides, and itemsize of F layout?

    Note: The code is usable as a numba-compiled function
    """
    nd = len(dims)
    # Check and skip 1s or 0s in inner dims
    firstax = 0
    while firstax < nd and dims[firstax] <= 1:
        firstax += 1

    # Early exit if all axis are 1s or 0s
    if firstax >= nd:
        return True

    # Check itemsize matches innermost stride
    if itemsize != strides[firstax]:
        return False

    # Check and skip 1s or 0s in outer dims
    lastax = nd - 1
    while lastax > firstax and dims[lastax] <= 1:
        lastax -= 1

    # Check remaining strides to be contiguous
    ax = firstax
    while ax < lastax:
        if strides[ax] * dims[ax] != strides[ax + 1]:
            return False
        ax += 1
    return True
