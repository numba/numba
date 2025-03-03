"""
Helper classes / mixins for defining types.
"""

import struct

from .abstract import ArrayCompatible, Dummy, IterableType, IteratorType
from numba.core.errors import NumbaTypeError, NumbaValueError


def validate_alignment(alignment: int):
    """
    Ensures that *alignment*, if not None, is a) greater than zero, b) a power
    of two, and c) a multiple of the size of a pointer.  If any of these
    conditions are not met, a NumbaValueError is raised.  Otherwise, this
    function returns None, indicating that the alignment is valid.
    """
    if alignment is None:
        return
    if not isinstance(alignment, int):
        raise NumbaValueError("Alignment must be an integer")
    if alignment <= 0:
        raise NumbaValueError("Alignment must be positive")
    if (alignment & (alignment - 1)) != 0:
        raise NumbaValueError("Alignment must be a power of 2")
    pointer_size = struct.calcsize("P")
    if (alignment % pointer_size) != 0:
        msg = f"Alignment must be a multiple of {pointer_size}"
        raise NumbaValueError(msg)


class Opaque(Dummy):
    """
    A type that is a opaque pointer.
    """


class SimpleIterableType(IterableType):

    def __init__(self, name, iterator_type):
        self._iterator_type = iterator_type
        super(SimpleIterableType, self).__init__(name)

    @property
    def iterator_type(self):
        return self._iterator_type


class SimpleIteratorType(IteratorType):

    def __init__(self, name, yield_type):
        self._yield_type = yield_type
        super(SimpleIteratorType, self).__init__(name)

    @property
    def yield_type(self):
        return self._yield_type


class Buffer(IterableType, ArrayCompatible):
    """
    Type class for objects providing the buffer protocol.
    Derived classes exist for more specific cases.
    """
    mutable = True
    slice_is_copy = False
    aligned = True

    # CS and FS are not reserved for inner contig but strided
    LAYOUTS = frozenset(['C', 'F', 'CS', 'FS', 'A'])

    def __init__(self, dtype, ndim, layout, readonly=False, name=None,
                 alignment=None):
        from .misc import unliteral, NoneType

        if isinstance(dtype, Buffer):
            msg = ("The dtype of a Buffer type cannot itself be a Buffer type, "
                   "this is unsupported behaviour."
                   "\nThe dtype requested for the unsupported Buffer was: {}.")
            raise NumbaTypeError(msg.format(dtype))
        if layout not in self.LAYOUTS:
            raise NumbaValueError("Invalid layout '%s'" % layout)
        if alignment is not None and not isinstance(alignment, NoneType):
            if not isinstance(alignment, int):
                alignment = alignment.literal_value
            validate_alignment(alignment)
        else:
            alignment = None
        self.dtype = unliteral(dtype)
        self.ndim = ndim
        self.layout = layout
        self.alignment = alignment
        if readonly:
            self.mutable = False
        if name is None:
            type_name = self.__class__.__name__.lower()
            if readonly:
                type_name = "readonly %s" % type_name
            name = "%s(%s, %sd, %s" % (type_name, dtype, ndim, layout)
            if alignment is not None:
                name += ", alignment=%d" % alignment
            name += ")"
        super(Buffer, self).__init__(name)

    @property
    def iterator_type(self):
        from .iterators import ArrayIterator
        return ArrayIterator(self)

    @property
    def as_array(self):
        return self

    def copy(self, dtype=None, ndim=None, layout=None, alignment=None):
        if dtype is None:
            dtype = self.dtype
        if ndim is None:
            ndim = self.ndim
        if layout is None:
            layout = self.layout
        if alignment is None:
            alignment = self.alignment
        return self.__class__(dtype=dtype, ndim=ndim, layout=layout,
                              readonly=not self.mutable, alignment=alignment)

    @property
    def key(self):
        return self.dtype, self.ndim, self.layout, self.mutable, self.alignment

    @property
    def is_c_contig(self):
        return self.layout == 'C' or (self.ndim <= 1 and self.layout in 'CF')

    @property
    def is_f_contig(self):
        return self.layout == 'F' or (self.ndim <= 1 and self.layout in 'CF')

    @property
    def is_contig(self):
        return self.layout in 'CF'
