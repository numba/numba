from __future__ import absolute_import
import types
import ctypes
import numba.types
from numba.targets.registry import CPUTarget
from numba import njit


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
    __slots__ = ('_name', '_function')

    def __init__(self, name, function):
        self._name = name
        self._function = function

    def __get__(self, instance, owner):
        print(self, instance, owner)
        raise NotImplementedError


class _NativeData(object):
    def __init__(self, clsname, descriptors):
        fieldtypes = [(d._name, d._type) for d in descriptors]
        self._type = numba.types.Structure(clsname, fieldtypes)
        self._ref_type = numba.types.StructRef(self._type)
        ctx = CPUTarget.target_context
        llty = ctx.get_value_type(self._type)
        sizeof = ctx.get_abi_sizeof(llty)
        byteseq = (ctypes.c_byte * sizeof)()
        self._data = byteseq
        self._dataptr = ctypes.addressof(self._data)

    @property
    def data_pointer(self):
        return self._dataptr

    @property
    def numba_type(self):
        return self._ref_type


class PlainOldData(type):
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
        for name, func in methods:
            dct[name] = _MethodDescriptor(name, func)

        def ctor(self, **kwargs):
            assert not kwargs, "args at ctor not implemented"
            self.__numba__ = _NativeData(clsname, descriptors)

        dct['__init__'] = ctor

        return super(PlainOldData, cls).__new__(cls, clsname, parents, dct)


