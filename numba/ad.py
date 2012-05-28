
import __builtin__
import os
import sys
import inspect
import trace
import opcode

import numpy as np

from .utils import itercode


class CallVM(object):
    def __init__(self, func):
        print 'CallVM', func
        self.func = func
        self.fco = func.func_code
        self.names = self.fco.co_names
        self.varnames = self.fco.co_varnames
        self.constants = self.fco.co_consts
        self.costr = func.func_code.co_code
        self.argnames = self.fco.co_varnames[:self.fco.co_argcount]
        self.stack = []

    def call(self, args, kwargs):
        self.rval = None
        self._myglobals = {}
        for name in self.names:
            print 'name', name
            try:
                self._myglobals[name] = self.func.func_globals[name]
            except KeyError:
                try:
                    self._myglobals[name] = __builtin__.__getattribute__(name)
                except AttributeError:
                    print 'WARNING: name lookup failed', name

        self._locals = [None]*len(self.fco.co_varnames)
        for i, name in enumerate(self.argnames):
            #print 'i', args, self.argnames, self.fco.co_varnames
            self._locals[i] = args[i]

        for i, op, arg in itercode(self.costr):
            name = opcode.opname[op]
            print 'OP: ', name
            getattr(self, 'op_' + name)(i, op, arg)

        return self.rval

    def op_BINARY_ADD(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        self.stack.append(arg1 + arg2)

    def op_BINARY_MULTIPLY(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        self.stack.append(arg1 * arg2)

    def op_CALL_FUNCTION(self, i, op, arg):
        # XXX: does this work with kwargs?
        args = [self.stack[-ii] for ii in range(arg, 0, -1)]
        if arg > 0:
            self.stack = self.stack[:-arg]
        func = self.stack.pop(-1)
        recurse = True
        if func.__module__ and func.__module__.startswith('numpy'):
            recurse = False
        print str(func)
        if 'built-in method' in str(func):
            recurse = False
        if recurse:
            vm = CallVM(func)
            rval = vm.call(args, {})
        else:
            rval = func(*args)
        self.stack.append(rval)

    def op_LOAD_GLOBAL(self, i, op, arg):
        print 'LOAD_GLOBAL', self.names[arg]
        self.stack.append(self._myglobals[self.names[arg]])

    def op_LOAD_ATTR(self, i, op, arg):
        print 'LOAD_ATTR', self.names[arg]
        TOS = self.stack[-1]
        self.stack[-1] = getattr(TOS, self.names[arg])

    def op_LOAD_CONST(self, i, op, arg):
        print 'LOAD_CONST', self.constants[arg]
        self.stack.append(self.constants[arg])

    def op_LOAD_FAST(self, i, op, arg):
        print 'LOAD_FAST', self.varnames[arg]
        self.stack.append(self._locals[arg])

    def op_STORE_FAST(self, i, op, arg):
        print 'STORE_FAST', self.varnames[arg]
        self._locals[arg] = self.stack.pop(-1)

    def op_RETURN_VALUE(self, i, op, arg):
        self.rval = self.stack.pop(-1)


class Watcher(object):
    def __init__(self, inputs):
        self.inputs = inputs

    def call(self, fn, *args, **kwargs):
        vm = CallVM(fn)
        return vm.call(args, kwargs)

