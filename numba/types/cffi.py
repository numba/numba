"""
Types associated with cffi support
"""
import re
from types import BuiltinFunctionType

from numba.typing.cffi_utils import get_func_pointer, get_struct_pointer
from numba.typeconv import Conversion
from numba import types


class CFFILibraryType(types.Opaque):
    def __init__(self, lib):
        self._func_names = set(
            f for f in dir(lib) if isinstance(getattr(lib, f), BuiltinFunctionType)
        )
        self._lib_name = re.match(r"<Lib object for '([^']+)'>", str(lib)).group(1)
        name = "cffi_lib<{}>".format(self._lib_name)
        super(CFFILibraryType, self).__init__(name)

    def has_func(self, func_name):
        return func_name in self._func_names

    def get_func_pointer(self, func_name):
        if func_name not in self._func_names:
            raise AttributeError(
                "Function {} is not present in the library {}".format(
                    func_name, self._lib_name
                )
            )
        return get_func_pointer(func_name)


class FFIType(types.Opaque):
    def __init__(self, ffi):
        self.ffi = ffi
        name = "FFI#{}".format(hex(id(ffi)))
        super(FFIType, self).__init__(name)

    def get_call_signatures(self):
        return [], True


class CFFIStructInstanceType(types.Type):
    def __init__(self, cffi_type):
        self.cffi_type = cffi_type
        name = "<instance> (" + self.cffi_type.cname + ")"
        self.struct = {}
        super(CFFIStructInstanceType, self).__init__(name)

    def can_convert_to(self, typingctx, other):
        if other == types.voidptr:
            return Conversion.safe

    def can_convert_from(self, typeingctx, other):
        if other == types.voidptr:
            return Conversion.safe


class CFFIPointer(types.CPointer):
    def __init__(self, dtype, owning=False):
        super(CFFIPointer, self).__init__(dtype)
        self.owning = owning
        owning_str = "(Owning)" if self.owning else ""
        self.name = "<ffi>(" + self.name + "*)" + owning_str

    @property
    def key(self):
        return (self.dtype, self.owning)

    @staticmethod
    def get_pointer(struct_ptr):
        return get_struct_pointer(struct_ptr)

    def __repr__(self):
        return self.name


class CFFINullPtrType(types.CPointer):
    def __init__(self):
        super(CFFINullPtrType, self).__init__(types.void)

    def can_convert_from(self, typeingctx, other):
        if isinstance(other, types.CFFIPointer):
            return Conversion.safe

    def can_convert_to(self, typeingctx, other):
        if isinstance(other, types.CFFIPointer):
            return Conversion.safe


class CFFIArrayType(CFFIPointer, types.Sequence):
    def __init__(self, dtype, length, owning=False):
        super(CFFIArrayType, self).__init__(dtype, owning)
        owning_str = "(Owning)" if self.owning else ""
        self.length = length
        self.name = "{}[{}]".format(self.dtype.name, self.length) + owning_str

    @property
    def iterator_type(self):
        return CFFIIteratorType(self)

    @property
    def yield_type(self):
        return CFFIStructRefType(self)


class CFFIIteratorType(types.BaseContainerIterator):
    container_class = CFFIArrayType

    def __init__(self, container):
        assert isinstance(container, self.container_class), container
        self.container = container
        yield_type = container.yield_type
        name = "iter(%s)" % container
        super(types.BaseContainerIterator, self).__init__(name, yield_type)

    def unify(self, typingctx, other):
        cls = type(self)
        if isinstance(other, cls):
            container = typingctx.unify_pairs(self.container, other.container)
            if container is not None:
                return cls(container)

    @property
    def key(self):
        return self.container


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
