# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.
#
# Provides a Descriptor for class definition to protect create constant fields
# and fields with constrains.
#
# NOTE: Originally part of my personal python script library PyON, which has not
# been distributed. This file is copied from my PyON project.
#

instanceof = lambda T: lambda X: isinstance(X, T)

class Descriptor(object):
    __slots__ = 'constant'

    def __new__(cls, **kwargs):
        if not __debug__:
            return object.__new__(Descriptor)

        constraints = kwargs.get('constraints')
        if constraints is not None:
            return object.__new__(ConstrainedDescriptor)
        else:
            return object.__new__(Descriptor)

    def __init__(self, **kwargs):
        if __debug__:
            if kwargs.get('constant'):
                # value is allowed to modify once.
                self.constant = True
            else:
                self.constant = False

    def __get__(self, obj, objtype=None):
        try:
            value = obj.__dict__[self]
        except KeyError:
            raise AttributeError('Descriptor is not defined.')
        else:
            return self.on_access(obj, value)


    def __set__(self, obj, value):
        value = self.on_change(obj, value)
        obj.__dict__[self] = value
        return value

    def __delete__(self, obj):
        del obj.__dict__[self]

    def on_access(self, obj, value):
        '''Called when the value is being accessed.
        The owner will receive the returned value.
        This method is free to modify the value before returning.
        '''
        return value

    def on_change(self, obj, value):
        '''Called when the value is being modified.
        The owner is trying to save a value.
        This method can provide value checking or modification.
        The returned value is the final value being stored.
        '''
        if __debug__ and self.constant:
            name = 'rdonly_%d'%id(self)
            if name in obj.__dict__:
                raise TypeError('Value in instance %(obj)s is marked constant.'%locals())
            else:
                # Modify only once
                obj.__dict__[name]=None
        return value


class ConstrainedDescriptor(Descriptor):
    '''Do not call this directly. Use the Descriptor parent class for
    constructor. The __new__() method will automatically dispatch
    the construction to the correct class implementation.

    Constaints is disabled if __debug__ is False.
    '''
    __slots__ = 'constraints'

    def __init__(self, constraints, **kwargs):
        # Invoke parent constructor
        super(ConstrainedDescriptor, self).__init__(**kwargs)
        # Perform initialization for self
        if callable(constraints):
            constraints = [constraints]
        self.constraints = list(constraints)
        # ensure all constraints are callable
        for fn in self.constraints:
            assert callable(fn)

    def on_change(self, obj, value):
        value = super(ConstrainedDescriptor, self).on_change(obj, value)
        # Verify value against all constraints
        if all( fn(value) for fn in self.constraints ):
            return value
        else:
            raise ValueError("Constraint error in object %(obj)s: %(value)s"%locals())


