from __future__ import print_function, absolute_import
from . import _typeconv


class TypeManager(object):
    def __init__(self):
        self._ptr = _typeconv.new_type_manager()

    def get(self, typ):
        return _typeconv.get_type(self._ptr, typ)

    def select_overload(self, sig, overloads):
        return _typeconv.select_overload(self._ptr, sig, overloads)

    def check_compatible(self, fromty, toty):
        return _typeconv.check_compatible(self._ptr, fromty, toty)

    def set_compatbile(self, fromty, toty, by):
        _typeconv.set_compatible(self._ptr, fromty, toty, by)

    def set_promote(self, fromty, toty):
        self.set_compatbile(fromty, toty, ord("p"))

    def set_unsafe_convert(self, fromty, toty):
        self.set_compatbile(fromty, toty, ord("u"))

    def set_safe_convert(self, fromty, toty):
        self.set_compatbile(fromty, toty, ord("s"))

    def get_pointer(self):
        return _typeconv.get_pointer(self._ptr)
