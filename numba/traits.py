"""
Minimal traits implementation:

    @traits
    class MyClass(object):

        attr = Instance(SomeClass)
        my_delegation = Delegate('attr')
"""

import inspect

# from numba.utils import TypedProperty

def traits(cls):
    "@traits class decorator"
    for name, py_func in vars(cls).items():
        if isinstance(py_func, TraitBase):
            py_func.set_attr_name(name)

    return cls

class TraitBase(object):
    "Base class for traits"

    def __init__(self, value, doc=None):
        self.value = value
        self.doc = doc

    def set_attr_name(self, name):
        self.attr_name = name

class Delegate(TraitBase):
    """
    Delegate to some other object.
    """

    def __init__(self, value, delegate_attr_name=None, doc=None):
        super(Delegate, self).__init__(value, doc=doc)
        self.delegate_attr_name = delegate_attr_name

    def obj(self, instance):
        return getattr(instance, self.value)

    @property
    def attr(self):
        return self.delegate_attr_name or self.attr_name

    def __get__(self, instance, owner):
        return getattr(self.obj(instance), self.attr)

    def __set__(self, instance, value):
        return setattr(self.obj(instance), self.attr, value)

    def __delete__(self, instance):
        delattr(self.obj(instance), self.attr)