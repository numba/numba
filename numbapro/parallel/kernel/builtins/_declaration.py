from collections import namedtuple

impl_record = namedtuple('impl_record', ['name', 'impl'])

class Declaration(object):
    def __new__(cls, name, doc):
        newcls = type(name, (cls,), {'__doc__' : doc})
        return object.__new__(newcls)

    def __init__(self, name, doc):
        self.__impls = {}

    def register(self, target, name=None):
        def _register(impl):
            self.__impls[target] = impl_record(name, impl)
            return impl
        return _register

    def get_implementation(self, target):
        return self.__impls[target]

class Configuration(Declaration):
    pass
