from __future__ import print_function, absolute_import
from . import _typeconv


class TypeManager(object):
    def __init__(self):
        self._ptr = _typeconv.new_type_manager()

    def select_overload(self, sig, overloads):
        sig = [t._code for t in sig]
        overloads = [[t._code for t in s] for s in overloads ]
        return _typeconv.select_overload(self._ptr, sig, overloads)

    def check_compatible(self, fromty, toty):
        return _typeconv.check_compatible(self._ptr, fromty._code, toty._code)

    def set_compatbile(self, fromty, toty, by):
        _typeconv.set_compatible(self._ptr, fromty._code, toty._code, by)

    def set_promote(self, fromty, toty):
        self.set_compatbile(fromty, toty, ord("p"))

    def set_unsafe_convert(self, fromty, toty):
        self.set_compatbile(fromty, toty, ord("u"))

    def set_safe_convert(self, fromty, toty):
        self.set_compatbile(fromty, toty, ord("s"))

    def get_pointer(self):
        return _typeconv.get_pointer(self._ptr)
