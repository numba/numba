"""
Implement logic relating to wrapping (box) and unwrapping (unbox) instances
of jitclasses for use inside the python interpreter.
"""
from __future__ import print_function, absolute_import

import inspect
from functools import wraps, partial

from numba import types, cgutils
from numba.pythonapi import box, unbox, NativeValue
from numba import njit
from numba.six import exec_
from ._box import Box


_getter_code_template = """
def accessor(__numba_self_):
    return __numba_self_.{0}
"""

_setter_code_template = """
def mutator(__numba_self_, __numba_val):
    __numba_self_.{0} = __numba_val
"""

_method_code_template = """
def method(__numba_self_, {args}):
    return __numba_self_.{method}({args})
"""


def _generate_property(field, template, fname):
    """
    Generate simple function that get/set a field of the instance
    """
    source = template.format(field)
    glbls = {}
    exec_(source, glbls)
    return njit(glbls[fname])


_generate_getter = partial(_generate_property, template=_getter_code_template,
                           fname='accessor')
_generate_setter = partial(_generate_property, template=_setter_code_template,
                           fname='mutator')


def _generate_method(name, func):
    """
    Generate a wrapper for calling a method
    """
    argspec = inspect.getargspec(func)
    assert not argspec.varargs, 'varargs not supported'
    assert not argspec.keywords, 'keywords not supported'
    assert not argspec.defaults, 'defaults not supported'

    args = ', '.join(argspec.args[1:])  # skipped self arg
    source = _method_code_template.format(method=name, args=args)
    glbls = {}
    exec_(source, glbls)
    method = njit(glbls['method'])

    @wraps(func)
    def wrapper(*args, **kwargs):
        return method(*args, **kwargs)

    return wrapper


_cache_specialized_box = {}


def _specialize_box(typ):
    """
    Create a subclass of Box that is specialized to the jitclass.

    This function caches the result to avoid code bloat.
    """
    # Check cache
    if typ in _cache_specialized_box:
        return _cache_specialized_box[typ]
    dct = {'__slots__': (),
           '_numba_type_': typ}
    # Inject attributes as class properties
    for field in typ.struct:
        if not field.startswith('_'):
            getter = _generate_getter(field)
            setter = _generate_setter(field)
            dct[field] = property(getter, setter)
    # Inject properties as class properties
    for field, impdct in typ.jitprops.items():
        if not field.startswith('_'):
            getter = None
            setter = None
            if 'get' in impdct:
                getter = _generate_getter(field)
            if 'set' in impdct:
                setter = _generate_setter(field)
            dct[field] = property(getter, setter)
    # Inject methods as class members
    for name, func in typ.methods.items():
        if not name.startswith('_'):
            getter = _generate_method(name, func)
            dct[name] = getter
    # Create subclass
    subcls = type(typ.classname, (Box,), dct)
    # Store to cache
    _cache_specialized_box[typ] = subcls

    return subcls


###############################################################################
# Implement box/unbox for call wrapper

@box(types.ClassInstanceType)
def _box_class_instance(typ, val, c):
    meminfo, dataptr = cgutils.unpack_tuple(c.builder, val)

    lluintp = c.context.get_data_type(types.uintp)

    addr_meminfo = c.pyapi.from_native_value(types.uintp,
                                             c.builder.ptrtoint(meminfo,
                                                                lluintp))
    addr_dataptr = c.pyapi.from_native_value(types.uintp,
                                             c.builder.ptrtoint(dataptr,
                                                                lluintp))

    box_subclassed = _specialize_box(typ)
    # Note: the ``box_subclassed`` is kept alive by the cache
    int_addr_boxcls = c.context.get_constant(types.uintp, id(box_subclassed))

    box_cls = c.builder.inttoptr(int_addr_boxcls, c.pyapi.pyobj)

    args = [addr_meminfo, addr_dataptr]
    res = c.pyapi.call_function_objargs(box_cls, args)

    # Clean up
    c.pyapi.decref(addr_meminfo)
    c.pyapi.decref(addr_dataptr)

    return res


@unbox(types.ClassInstanceType)
def _unbox_class_instance(typ, val, c):
    struct_cls = cgutils.create_struct_proxy(typ)
    inst = struct_cls(c.context, c.builder)

    int_meminfo = c.pyapi.object_getattr_string(val, "_meminfoptr")
    int_dataptr = c.pyapi.object_getattr_string(val, "_dataptr")

    ptr_meminfo = c.pyapi.long_as_voidptr(int_meminfo)
    ptr_dataptr = c.pyapi.long_as_voidptr(int_dataptr)

    c.pyapi.decref(int_meminfo)
    c.pyapi.decref(int_dataptr)

    inst.meminfo = c.builder.bitcast(ptr_meminfo, inst.meminfo.type)
    inst.data = c.builder.bitcast(ptr_dataptr, inst.data.type)

    ret = inst._getvalue()

    c.context.nrt_incref(c.builder, typ, ret)

    return NativeValue(ret, is_error=c.pyapi.c_api_error())
