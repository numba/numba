"""
Minimal traits implementation:

    @traits
    class MyClass(object):

        attr = Instance(SomeClass)
        my_delegation = Delegate('attr')
"""

import inspect

from numba.utils import TypedProperty

def traits(cls):
    "@traits class decorator"
    for name, py_func in vars(cls).iteritems():
        if isinstance(py_func, TraitBase):
            py_func.set_attr_name(name)

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

    def __init__(self, value, delegate_method_name=None, doc=None):
        super(Delegate, self).__init__(value, doc=doc)
        self.delegate_attr_name = delegate_method_name

    def set_attr_name(self, name):
        if self.delegate_attr_name is None:
            self.delegate_attr_name = name

    def __get__(self, instance, owner):
        return getattr(instance, self.delegate_attr_name)

    def __set__(self, instance, value):
        return getattr(instance, self.delegate_attr_name, value)
