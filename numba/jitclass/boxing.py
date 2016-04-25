"""
Implement logic relating to wrapping (box) and unwrapping (unbox) instances
of jitclasses for use inside the python interpreter.
"""
from __future__ import print_function, absolute_import

import inspect
from functools import wraps, partial

from llvmlite import ir

from numba import types, cgutils
from numba.pythonapi import box, unbox, NativeValue
from numba import njit
from numba.six import exec_
from . import _box


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


_ignored_special_methods = set(['__init__'])


def _handle_special_methods(dct, name, func):
    if name in _ignored_special_methods:
        return
    elif name in ['__hash__', '__eq__']:
        dct[name] = _generate_method(name, func)
    else:
        msg = "unsupported special method: {0}".format(name)
        raise NotImplementedError(msg)


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
        getter = _generate_getter(field)
        setter = _generate_setter(field)
        dct[field] = property(getter, setter)
    # Inject properties as class properties
    for field, impdct in typ.jitprops.items():
        if not field.startswith('__'):
            getter = None
            setter = None
            if 'get' in impdct:
                getter = _generate_getter(field)
            if 'set' in impdct:
                setter = _generate_setter(field)
            dct[field] = property(getter, setter)
    # Inject methods as class members
    for name, func in typ.methods.items():
        if not (name.startswith('__') and name.endswith('__')):
            dct[name] = _generate_method(name, func)
        else:
            _handle_special_methods(dct, name, func)

    # Create subclass
    subcls = type(typ.classname, (_box.Box,), dct)
    # Store to cache
    _cache_specialized_box[typ] = subcls

    # Pre-compile attribute getter.
    # Note: This must be done after the "box" class is created because
    #       compiling the getter requires the "box" class to be defined.
    for k, v in dct.items():
        if isinstance(v, property):
            prop = getattr(subcls, k)
            if prop.fget is not None:
                fget = prop.fget
                fast_fget = fget.compile((typ,))
                fget.disable_compile()
                setattr(subcls, k,
                        property(fast_fget, prop.fset, prop.fdel))

    return subcls


###############################################################################
# Implement box/unbox for call wrapper

@box(types.ClassInstanceType)
def _box_class_instance(typ, val, c):
    meminfo, dataptr = cgutils.unpack_tuple(c.builder, val)

    # Create Box instance
    box_subclassed = _specialize_box(typ)
    # Note: the ``box_subclassed`` is kept alive by the cache
    int_addr_boxcls = c.context.get_constant(types.uintp, id(box_subclassed))

    box_cls = c.builder.inttoptr(int_addr_boxcls, c.pyapi.pyobj)
    box = c.pyapi.call_function_objargs(box_cls, ())

    # Initialize Box instance
    llvoidptr = ir.IntType(8).as_pointer()
    addr_meminfo = c.builder.bitcast(meminfo, llvoidptr)
    addr_data = c.builder.bitcast(dataptr, llvoidptr)

    def set_member(member_offset, value):
        # Access member by byte offset
        offset = c.context.get_constant(types.uint32, member_offset)
        byte_ptr = c.builder.bitcast(box, llvoidptr)
        ptr = c.builder.gep(byte_ptr, [offset])
        casted = c.builder.bitcast(ptr, llvoidptr.as_pointer())
        c.builder.store(value, casted)

    set_member(_box.box_meminfoptr_offset, addr_meminfo)
    set_member(_box.box_dataptr_offset, addr_data)
    return box


@unbox(types.ClassInstanceType)
def _unbox_class_instance(typ, val, c):
    def access_member(member_offset):
        # Access member by byte offset
        offset = c.context.get_constant(types.uint32, member_offset)
        llvoidptr = ir.IntType(8).as_pointer()
        byte_ptr = c.builder.bitcast(val, llvoidptr)
        ptr = c.builder.gep(byte_ptr, [offset])
        casted = c.builder.bitcast(ptr, llvoidptr.as_pointer())
        return c.builder.load(casted)

    struct_cls = cgutils.create_struct_proxy(typ)
    inst = struct_cls(c.context, c.builder)

    # load from Python object
    ptr_meminfo = access_member(_box.box_meminfoptr_offset)
    ptr_dataptr = access_member(_box.box_dataptr_offset)

    # store to native structure
    inst.meminfo = c.builder.bitcast(ptr_meminfo, inst.meminfo.type)
    inst.data = c.builder.bitcast(ptr_dataptr, inst.data.type)

    ret = inst._getvalue()

    c.context.nrt_incref(c.builder, typ, ret)

    return NativeValue(ret, is_error=c.pyapi.c_api_error())
