"""
Utility function and type cache for cffi support
"""

from functools import partial
from types import BuiltinFunctionType

from numba import numpy_support
from numba import types
from numba.core.typing import templates
from numba import cgutils


class CFFITypeInfo(object):
    """
    Cache cffi type info
    """

    def __init__(self, ffi, cffi_type):
        self.cname = cffi_type.cname
        self.cffi_type = cffi_type
        self.cffi_ptr_t = ffi.typeof(self.cffi_type.cname + "*")


class CFFIStructTypeCache(object):
    def __init__(self):
        self.ctypes_cache = {}

    def add_type(self, mod, cffi_type):
        self.ctypes_cache[hash(cffi_type)] = CFFITypeInfo(mod.ffi, cffi_type)

    def get_type_by_hash(self, h):
        return self.ctypes_cache[h]

    def get_type_hash(self, cffi_type):
        return hash(cffi_type)


cffi_types_cache = CFFIStructTypeCache()

_cached_type_map = None


def cffi_type_map():
    """
    Lazily compute type map, as calling ffi.typeof() involves costly
    parsing of C code...
    """
    global _cached_type_map
    if _cached_type_map is None:
        _cached_type_map = {
            ffi.typeof("bool"): types.boolean,
            ffi.typeof("char"): types.char,
            ffi.typeof("short"): types.short,
            ffi.typeof("int"): types.intc,
            ffi.typeof("long"): types.long_,
            ffi.typeof("long long"): types.longlong,
            ffi.typeof("unsigned char"): types.uchar,
            ffi.typeof("unsigned short"): types.ushort,
            ffi.typeof("unsigned int"): types.uintc,
            ffi.typeof("unsigned long"): types.ulong,
            ffi.typeof("unsigned long long"): types.ulonglong,
            ffi.typeof("int8_t"): types.char,
            ffi.typeof("uint8_t"): types.uchar,
            ffi.typeof("int16_t"): types.short,
            ffi.typeof("uint16_t"): types.ushort,
            ffi.typeof("int32_t"): types.intc,
            ffi.typeof("uint32_t"): types.uintc,
            ffi.typeof("int64_t"): types.longlong,
            ffi.typeof("uint64_t"): types.ulonglong,
            ffi.typeof("float"): types.float_,
            ffi.typeof("double"): types.double,
            ffi.typeof("ssize_t"): types.intp,
            ffi.typeof("size_t"): types.uintp,
            ffi.typeof("void"): types.void,
        }
    return _cached_type_map


_ool_func_ptr = {}
_ool_libraries = set()
_ool_func_types = {}
_ool_struct_types = {}

_ffi_instances = set()

try:
    import cffi

    ffi = cffi.FFI()
except ImportError:
    ffi = None

SUPPORTED = ffi is not None


def is_ffi_lib(lib):
    # we register libs on register_module call
    return lib in _ool_libraries


def is_ffi_instance(obj):
    # Compiled FFI modules have a member, ffi, which is an instance of
    # CompiledFFI, which behaves similarly to an instance of cffi.FFI. In
    # order to simplify handling a CompiledFFI object, we treat them as
    # if they're cffi.FFI instances for typing and lowering purposes.
    try:
        return obj in _ffi_instances or isinstance(obj, cffi.FFI)
    except TypeError:  # Unhashable type possible
        return False


def is_cffi_func(obj):
    """Check whether the obj is a CFFI function"""
    try:
        return ffi.typeof(obj).kind == "function"
    except TypeError:
        try:
            return obj.__name__ in _ool_func_types
        except:
            return False


def is_cffi_struct(obj):
    """
    Check whether the obj is a CFFI struct or a pointer to one
    """
    try:
        t = ffi.typeof(obj)
    except TypeError:
        return False
    if t.kind == "pointer":
        return t.item.kind == "struct"
    elif t.kind == "struct":
        return True
    else:
        return False


def get_func_pointer(cffi_func):
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """
    if isinstance(cffi_func, str):
        func_name = cffi_func
    else:
        func_name = cffi_func.__name__
    if func_name in _ool_func_ptr:
        return _ool_func_ptr[func_name]
    return int(ffi.cast("uintptr_t", cffi_func))


def get_struct_pointer(cffi_struct_ptr):
    """
    Convert struct pointer to integer
    """
    return int(ffi.cast("uintptr_t", cffi_struct_ptr))


def map_struct_to_numba_type(cffi_type):
    """
    Convert a cffi type to a numba StructType
    """
    if cffi_type in _ool_struct_types:
        return _ool_struct_types[cffi_type]

    # forward declare the struct type
    forward = types.deferred_type()
    _ool_struct_types[cffi_type] = forward
    struct_type = types.CFFIStructInstanceType(cffi_type)
    for k, v in cffi_type.fields:
        if v.bitshift != -1:
            msg = "field {!r} has bitshift, this is not supported"
            raise ValueError(msg.format(k))
        if v.flags != 0:
            msg = "field {!r} has flags, this is not supported"
            raise ValueError(msg.format(k))
        if v.bitsize != -1:
            msg = "field {!r} has bitsize, this is not supported"
            raise ValueError(msg.format(k))
        struct_type.struct[k] = map_type(v.type, use_record_dtype=False)
    forward.define(struct_type)
    _ool_struct_types[cffi_type] = struct_type
    return struct_type


def map_struct_to_record_dtype(cffi_type):
    """Convert a cffi type into a NumPy Record dtype
    """
    fields = {
        "names": [],
        "formats": [],
        "offsets": [],
        "itemsize": ffi.sizeof(cffi_type),
    }
    is_aligned = True
    for k, v in cffi_type.fields:
        # guard unsupport values
        if v.bitshift != -1:
            msg = "field {!r} has bitshift, this is not supported"
            raise ValueError(msg.format(k))
        if v.flags != 0:
            msg = "field {!r} has flags, this is not supported"
            raise ValueError(msg.format(k))
        if v.bitsize != -1:
            msg = "field {!r} has bitsize, this is not supported"
            raise ValueError(msg.format(k))
        dtype = numpy_support.as_dtype(map_type(v.type, use_record_dtype=True))
        fields["names"].append(k)
        fields["formats"].append(dtype)
        fields["offsets"].append(v.offset)
        # Check alignment
        is_aligned &= v.offset % dtype.alignment == 0

    return numpy_support.from_dtype(np.dtype(fields, align=is_aligned))


def make_function_type(cffi_func, use_record_dtype=False):
    """
    Return a Numba type for the given CFFI function pointer.
    """
    if isinstance(cffi_func, str):
        cffi_type = _ool_func_types.get(cffi_func)
    else:
        cffi_type = _ool_func_types.get(cffi_func.__name__) or ffi.typeof(cffi_func)
    sig = map_type(cffi_type, use_record_dtype=use_record_dtype)
    return types.ExternalFunctionPointer(sig, get_pointer=get_func_pointer)


def get_struct_type(cffi_struct):
    """
    Return a Numba type for the given CFFI struct
    """
    t = ffi.typeof(cffi_struct)
    if t.kind == "pointer":
        return types.CFFIPointer(cffi_type_map()[t.item])

    return cffi_type_map()[t]


_free_ffi = None


def get_free_ffi():
    """We use it to resolve circular dependecies"""
    global _free_ffi
    if _free_ffi is None:
        from numba import extending, njit

        def free_ffi_new(typingctx, cffi_ptr):
            if isinstance(cffi_ptr, types.CFFIPointer):
                sig = templates.signature(types.void, cffi_ptr)

                def codegen(context, builder, signature, args):
                    ptr = builder.bitcast(args[0], cgutils.voidptr_t)
                    context.nrt.free(builder, ptr)

                return sig, codegen

        free_intr = extending.intrinsic(free_ffi_new)

        def free_impl(cffi_ptr):
            free_intr(cffi_ptr)

        _free_ffi = njit(free_impl)
    return _free_ffi


def struct_from_ptr(hash_, intptr, owned):
    ret = ffi.cast(cffi_types_cache.get_type_by_hash(hash_).cffi_ptr_t, intptr)
    print("owned is", owned)
    if owned:
        print("Binding collector")
        ret = ffi.gc(ret, get_free_ffi())
    return ret


def map_type(cffi_type, use_record_dtype=False):
    """
    Map CFFI type to numba type.

    Parameters
    ----------
    cffi_type:
        The CFFI type to be converted.
    use_record_dtype: bool (default: False)
        When True, struct types are mapped to a NumPy Record dtype.

    """
    primed_map_type = partial(map_type, use_record_dtype=use_record_dtype)
    kind = getattr(cffi_type, "kind", "")
    result = cffi_type_map().get(cffi_type)
    if result is not None:
        return result
    if kind == "union":
        raise TypeError("No support for CFFI union")
    elif kind == "function":
        if cffi_type.ellipsis:
            raise TypeError("vararg function is not supported")
        restype = primed_map_type(cffi_type.result)
        argtypes = [primed_map_type(arg) for arg in cffi_type.args]
        return templates.signature(restype, *argtypes)
    elif kind == "pointer":
        pointee = cffi_type.item
        if pointee.kind == "void":
            return types.voidptr
        else:
            try:
                return types.CFFIPointer(primed_map_type(pointee))
            except TypeError:
                return types.voidptr
    elif kind == "array":
        dtype = primed_map_type(cffi_type.item)
        nelem = cffi_type.length
        return types.NestedArray(dtype=dtype, shape=(nelem,))
    elif kind == "struct":
        if use_record_dtype:
            return map_struct_to_record_dtype(cffi_type)
        else:
            return map_struct_to_numba_type(cffi_type)
    else:
        raise TypeError(cffi_type)


def register_type(cffi_type, numba_type):
    """
    Add typing for a given CFFI type to the typemap
    """
    tm = cffi_type_map()
    tm[cffi_type] = numba_type


def register_module(mod):
    """
    Add typing for all functions in an out-of-line CFFI module to the typemap
    """
    if mod.lib in _ool_libraries:
        # module already registered, don't do anything
        return
    _ool_libraries.add(mod.lib)
    for t in mod.ffi.list_types()[0]:
        cffi_type = mod.ffi.typeof(t)
        register_type(cffi_type, map_struct_to_numba_type(cffi_type))
        cffi_types_cache.add_type(mod, cffi_type)
    for f in dir(mod.lib):
        f = getattr(mod.lib, f)
        if isinstance(f, BuiltinFunctionType):
            _ool_func_types[f.__name__] = mod.ffi.typeof(f)
            addr = mod.ffi.addressof(mod.lib, f.__name__)
            _ool_func_ptr[f.__name__] = int(mod.ffi.cast("uintptr_t", addr))
        _ffi_instances.add(mod.ffi)

