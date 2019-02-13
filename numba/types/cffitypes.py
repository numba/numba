"""
Types associated with cffi support
"""
import re
from types import BuiltinFunctionType

from numba.typeconv import Conversion
from .abstract import Type
from .containers import Sequence, BaseContainerIterator
from .misc import Opaque, NoneType, CPointer, voidptr


class CFFILibraryType(Opaque):
    def __init__(self, lib):
        self._func_names = set(
            f for f in dir(lib) if isinstance(getattr(lib, f), BuiltinFunctionType)
        )
        self._lib_name = re.match(r"<Lib object for '([^']+)'>", str(lib)).group(1)
        name = "cffi_lib<{}>".format(self._lib_name)
        super(CFFILibraryType, self).__init__(name)

    def has_func(self, func_name):
        return func_name in self._func_names


class FFIType(Opaque):
    def __init__(self, ffi):
        self.ffi = ffi
        name = "FFI#{}".format(hex(id(ffi)))
        super(FFIType, self).__init__(name)

    def get_call_signatures(self):
        return [], True


class CFFIStructInstanceType(Type):
    def __init__(self, cffi_type):
        self.cffi_type = cffi_type
        name = "<instance> (" + self.cffi_type.cname + ")"
        self.struct = {}
        super(CFFIStructInstanceType, self).__init__(name)

    def can_convert_to(self, typingctx, other):
        if other == voidptr:
            return Conversion.safe

    def can_convert_from(self, typeingctx, other):
        if other == voidptr:
            return Conversion.safe


class CFFIPointer(CPointer):
    def __init__(self, dtype):
        super(CFFIPointer, self).__init__(dtype)
        self.name = "<ffi>(" + self.name + ")"

    @property
    def key(self):
        return self.dtype

    def __repr__(self):
        return self.name


class CFFINullPtrType(CPointer):
    def __init__(self):
        super(CFFINullPtrType, self).__init__(NoneType("nullptr"))

    def can_convert_from(self, typeingctx, other):
        if isinstance(other, CFFIPointer):
            return Conversion.safe

    def can_convert_to(self, typeingctx, other):
        if isinstance(other, CFFIPointer):
            return Conversion.safe


class CFFIArrayType(CFFIPointer, Sequence):
    def __init__(self, dtype, length):
        super(CFFIArrayType, self).__init__(dtype)
        self.length = length
        self.name = "{}[{}]".format(self.dtype.name, self.length)

    @property
    def iterator_type(self):
        return CFFIIteratorType(self)

    @property
    def yield_type(self):
        return CFFIStructRefType(self)


class CFFIIteratorType(BaseContainerIterator):
    container_class = CFFIArrayType

    def __init__(self, container):
        assert isinstance(container, self.container_class), container
        self.container = container
        yield_type = container.yield_type
        name = "iter(%s)" % container
        super(BaseContainerIterator, self).__init__(name, yield_type)

    def unify(self, typingctx, other):
        cls = type(self)
        if isinstance(other, cls):
            container = typingctx.unify_pairs(self.container, other.container)
            if container is not None:
                return cls(container)

    @property
    def key(self):
        return self.container


class CFFIOwningType(CFFIPointer):
    def __init__(self, *args, **kwargs):
        super(CFFIOwningType, self).__init__(*args, **kwargs)
        self.name = self.name + "(Owning)"

    @property
    def key(self):
        return (self.dtype, "owning")


class CFFIOwningPointerType(CFFIOwningType):
    def __init__(self, dtype):
        super(CFFIOwningPointerType, self).__init__(dtype)


class CFFIOwningArrayType(CFFIOwningType, CFFIArrayType):
    def __init__(self, dtype, length):
        super(CFFIOwningArrayType, self).__init__(dtype, length)


class CFFIStructRefType(CFFIPointer):
    def __init__(self, ptrtype):
        super(CFFIStructRefType, self).__init__(ptrtype.dtype)
        self.ptrtype = ptrtype
        self.name = "Ref#{}".format(self.dtype.name)

    def can_convert_to(self, typingctx, other):
        if isinstance(other, CFFIStructRefType) or isinstance(
            other, CFFIStructInstanceType
        ):
            return typingctx.can_convert(self.dtype, other)

    def can_convert_from(self, typingctx, other):
        if isinstance(other, CFFIStructRefType) or isinstance(
            other, CFFIStructInstanceType
        ):
            return typingctx.can_convert(self.dtype, other)

    def __repr__(self):
        return self.name
