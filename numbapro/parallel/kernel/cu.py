import math
import numpy
import numba
from collections import namedtuple

from .builtins._declaration import Declaration

Task = namedtuple('Task', ['func', 'ntid', 'args'])

class ComputeUnit(object):
    registered_targets = {}

    def __new__(self, target):
        cls = self.registered_targets[target]
        obj = object.__new__(cls)
        return obj

    def __init__(self, target):
        self.__target = target
        self.__state = State()
        self._init()

    @property
    def _state(self):
        return self.__state

    @property
    def target(self):
        return self.__target

    def configure(self, key, val):
        self._configure(key, val)

    def enqueue(self, kernel, ntid, args=()):
        if isinstance(kernel, Declaration):
            name, impl = kernel.get_implementation(self.target)
            self._execute_builtin(name, impl, ntid, args)
        else:
            self._execute_kernel(kernel, ntid, args)

    def wait(self):
        self._wait()

    def close(self):
        self._close()

    def input(self, ary):
        return self._input(ary)

    def output(self, ary):
        return self._output(ary)

    def inout(self, ary):
        return self._inout(ary)

    def scratch(self, shape, dtype=numpy.float, order='C'):
        return self._scratch(shape=shape, dtype=dtype, order=order)

    def scratch_like(self, ary):
        order = ''
        if ary.flags['C_CONTIGUOUS']:
            order = 'C'
        elif ary.flags['F_CONTIGUOUS']:
            order = 'F'
        return self.scratch(shape=ary.shape, dtype=ary.dtype, order=order)

    #
    # Methods to be overrided in subclass
    #

    def _init(self):
        '''Object initialization method.
        
        Default behavior does nothing.
        '''
        pass

    def _configure(self, key, val):
        key.get_implementation(self.target).impl(self, val)

    def _execute_builtin(name, impl, ntid, args):
        pass

    def _execute_kernel(self, func, ntid, ins, outs):
        raise NotImplementedError


    def _enqueue_write(self, ary):
        pass
    
    def _wait(self):
        pass

    def _close(self):
        pass


class State(object):
    __slots__ = '_store'
    def __init__(self):
        self._store = {}

    def get(self, name, default=None):
        return self._store.get(name, default)

    def set(self, name, value):
        setattr(self, name, value)

    def __getattr__(self, name):
        assert not name.startswith('_')
        return self._store_[name]

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(State, self).__setattr__(name, value)
        else:
            self._store[name] = value

# Short hand for compute unit
CU = ComputeUnit

