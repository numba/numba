from __future__ import absolute_import
import types
import ctypes
import numba.types
from functools import partial
from numba.targets.registry import CPUTarget
from numba import jit, njit
from numba import utils



_field_setter_name = "field_setter{clsname}_{field}"
_field_setter_template = """
def field_setter{clsname}_{field}(base, value):
    base.{field} = value
"""


def _build_field_setter(clsname, field):
    src = _field_setter_template.format(clsname=clsname, field=field)
    dct = {}
    exec(src, dct)
    fnobj = dct[_field_setter_name.format(clsname=clsname, field=field)]
    return njit(fnobj)


_field_getter_name = "field_getter{clsname}_{field}"
_field_getter_template = """
def field_getter{clsname}_{field}(base):
    return base.{field}
"""


def _build_field_getter(clsname, field):
    src = _field_getter_template.format(clsname=clsname, field=field)
    dct = {}
    exec(src, dct)
    fnobj = dct[_field_getter_name.format(clsname=clsname, field=field)]
    return njit(fnobj)


class _FieldDescriptor(object):
    __slots__ = ('_clsname', '_name', '_type', '_offset', '_setter', '_getter')

    def __init__(self, clsname, name, nbtype, offset):
        self._clsname = clsname
        self._name = name
        self._type = nbtype
        self._offset = offset
        self._setter = _build_field_setter(self._clsname, self._name)
        self._getter = _build_field_getter(self._clsname, self._name)

    def __get__(self, instance, owner):
        return self._getter(instance)

    def __set__(self, instance, value):
        self._setter(instance, value)


class _MethodDescriptor(object):
    __slots__ = ('_name', '_function', '_compiled')

    def __init__(self, name, function):
        self._name = name
        self._function = function
        self._compiled = jit(self._function)

    def __get__(self, instance, owner):
        return partial(self._compiled, instance)


class _NativeDataSpec(object):
    __slots__ = ('_type', '_ref_type', '_ctype')

    def __init__(self, clsname, descriptors, methoddescriptors):
        fieldtypes = [(d._name, d._type) for d in descriptors]
        methods = [(d._name, d._compiled) for d in methoddescriptors]
        self._type = numba.types.Structure(clsname, fieldtypes, methods)
        self._ref_type = numba.types.StructRef(self._type)
        ctx = CPUTarget.target_context
        llty = ctx.get_value_type(self._type)
        sizeof = ctx.get_abi_sizeof(llty)
        self._ctype = (ctypes.c_byte * sizeof)


class _NativeData(object):
    __slots__ = ('_spec', '_data', '_dataptr')

    def __init__(self, spec):
        self._spec = spec
        self._data = self._spec._ctype()
        self._dataptr = utils.longint(ctypes.addressof(self._data))

    @property
    def data_pointer(self):
        return self._dataptr

    @property
    def numba_type(self):
        return self._spec._ref_type


class PlainOldDataMeta(type):
    def __new__(cls, clsname, parents, dct):
        # Discover all the methods.  They are still functions at this point.
        methods = [(name, value) for name, value in dct.items()
                   if isinstance(value, types.FunctionType)]

        # Set __slots__ to empty to disable __dict__ and disallow adding
        # fields dynamically
        dct['__slots__'] = ('__numba__',)

        # Insert descriptor for each field
        descriptors = []
        for offset, (name, typ) in enumerate(dct.pop('__fields__')):
            desc = _FieldDescriptor(clsname, name, typ, offset)
            dct[name] = desc
            descriptors.append(desc)

        # Insert descriptor for methods:
        methodescriptors = []
        for name, func in methods:
            desc = _MethodDescriptor(name, func)
            dct[name] = desc
            methodescriptors.append(desc)

        # Generate the "spec" for creating the native data
        ndspec = _NativeDataSpec(clsname, descriptors, methodescriptors)

        def ctor(self, **kwargs):
            assert not kwargs, "args at ctor not implemented"
            self.__numba__ = _NativeData(ndspec)

        dct['__init__'] = ctor

        return super(PlainOldDataMeta, cls).__new__(cls, clsname, parents, dct)


# Make an inheritable class to hide the metaclass for (2+3) compatibility
PlainOldData = utils.with_metaclass(PlainOldDataMeta)
