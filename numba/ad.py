"""
Example of how to use byte-code execution technique to trace accesses to numpy arrays.

This file demonstrates two applications of this technique:
* optimize numpy computations for repeated calling
* provide automatic differentiation of procedural code

"""

import __builtin__
import os
import sys
import inspect
import trace
import opcode

import numpy as np
import theano

from .utils import itercode

# Opcode help: http://docs.python.org/library/dis.html

# XXX: support full calling convention for named args, *args and **kwargs


class FrameVM(object):
    """
    A Class for evaluating a code block of CPython bytecode,
    and tracking accesses to numpy arrays.

    """
    def __init__(self, watcher, func):
        print 'FrameVM', func
        self.watcher = watcher
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
            #print 'name', name
            try:
                self._myglobals[name] = self.func.func_globals[name]
            except KeyError:
                try:
                    self._myglobals[name] = __builtin__.__getattribute__(name)
                except AttributeError:
                    #print 'WARNING: name lookup failed', name
                    pass

        self._locals = [None] * len(self.fco.co_varnames)
        for i, name in enumerate(self.argnames):
            #print 'i', args, self.argnames, self.fco.co_varnames
            self._locals[i] = args[i]

        self.code_iter = itercode(self.costr)
        jmp = None
        while True:
            try:
                i, op, arg = self.code_iter.send(jmp)
            except StopIteration:
                break
            name = opcode.opname[op]
            #print 'OP: ', i, name
            jmp = getattr(self, 'op_' + name)(i, op, arg)

        return self.rval

    def op_BINARY_ADD(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        r = arg1 + arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.svars[id(r)] = s1 + s2
            #print 'added sym'

    def op_BINARY_SUBTRACT(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        r = arg1 - arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.svars[id(r)] = s1 - s2

    def op_BINARY_MULTIPLY(self, i, op, arg):
        arg2 = self.stack.pop(-1)
        arg1 = self.stack.pop(-1)
        r = arg1 * arg2
        self.stack.append(r)
        if (id(arg1) in self.watcher.svars
                or id(arg2) in self.watcher.svars):
            s1 = self.watcher.svars.get(id(arg1), arg1)
            s2 = self.watcher.svars.get(id(arg2), arg2)
            self.watcher.svars[id(r)] = s1 * s2
            #print 'mul sym', id(r)

    def op_CALL_FUNCTION(self, i, op, arg):
        # XXX: does this work with kwargs?
        args = [self.stack[-ii] for ii in range(arg, 0, -1)]
        if arg > 0:
            self.stack = self.stack[:-arg]
        func = self.stack.pop(-1)
        recurse = True

        if func.__module__ and func.__module__.startswith('numpy'):
            recurse = False
        if 'built-in' in str(func):
            recurse = False

        if recurse:
            vm = FrameVM(self.watcher, func)
            rval = vm.call(args, {})
        else:
            #print 'running built-in', func, func.__name__, args
            rval = func(*args)
            if any(id(a) in self.watcher.svars for a in args):
                sargs = [self.watcher.svars.get(id(a), a)
                        for a in args]
                if func.__name__ == 'sum':
                    #print 'sym sum', sargs
                    self.watcher.svars[id(rval)] = theano.tensor.sum(*sargs)
                else:
                    raise NotImplementedError(func)
        self.stack.append(rval)

    def op_COMPARE_OP(self, i, op, arg):
        opname = opcode.cmp_op[arg]
        left = self.stack.pop(-1)
        right = self.stack.pop(-1)
        if 0: pass
        elif opname == '==': self.stack.append(left == right)
        elif opname == '!=': self.stack.append(left != right)
        else:
            raise NotImplementedError('comparison: %s' % opname)


    def op_FOR_ITER(self, i, op, arg):
        # either push tos.next()
        # or pop tos and send (arg)
        tos = self.stack[-1]
        try:
            next = tos.next()
            print 'next', next
            self.stack.append(next)
        except StopIteration:
            self.stack.pop(-1)
            return ('rel', arg)

    def op_JUMP_ABSOLUTE(self, i, op, arg):
        print 'sending', arg
        return ('abs', arg)

    def op_JUMP_IF_TRUE(self, i, op, arg):
        tos = self.stack[-1]
        if tos:
            return ('rel', arg)

    def op_GET_ITER(self, i, op, arg):
        # replace tos -> iter(tos)
        tos = self.stack[-1]
        self.stack[-1] = iter(tos)
        if id(tos) in self.watcher.svars:
            raise NotImplementedError('iterator of watched value')

    def op_LOAD_GLOBAL(self, i, op, arg):
        #print 'LOAD_GLOBAL', self.names[arg]
        self.stack.append(self._myglobals[self.names[arg]])

    def op_LOAD_ATTR(self, i, op, arg):
        #print 'LOAD_ATTR', self.names[arg]
        TOS = self.stack[-1]
        self.stack[-1] = getattr(TOS, self.names[arg])

    def op_LOAD_CONST(self, i, op, arg):
        #print 'LOAD_CONST', self.constants[arg]
        self.stack.append(self.constants[arg])

    def op_LOAD_FAST(self, i, op, arg):
        #print 'LOAD_FAST', self.varnames[arg]
        self.stack.append(self._locals[arg])

    def op_POP_BLOCK(self, i, op, arg):
        print 'pop block, what to do?'

    def op_POP_TOP(self, i, op, arg):
        self.stack.pop(-1)

    def op_PRINT_ITEM(self, i, op, arg):
        print self.stack.pop(-1),

    def op_PRINT_NEWLINE(self, i, op, arg):
        print ''

    def op_SETUP_LOOP(self, i, op, arg):
        print 'SETUP_LOOP, what to do?'

    def op_STORE_FAST(self, i, op, arg):
        #print 'STORE_FAST', self.varnames[arg]
        self._locals[arg] = self.stack.pop(-1)

    def op_RAISE_VARARGS(self, i, op, arg):
        if 1 <= arg:
            exc = self.stack.pop(-1)
        if 2 <= arg:
            param = self.stack.pop(-1)
        if 3 <= arg:
            tb = self.stack.pop(-1)
        raise NotImplementedError('exception handling')

    def op_RETURN_VALUE(self, i, op, arg):
        self.rval = self.stack.pop(-1)


class Watcher(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.svars = {}
        for var in inputs:
            self.svars[id(var)] = theano.tensor.vector()

    def call(self, fn, *args, **kwargs):
        vm = FrameVM(self, fn)
        return vm.call(args, kwargs)

    def grad_fn(self, rval, ival):
        sy = self.svars[id(rval)]
        sx = self.svars[id(ival)]
        dydx = theano.tensor.grad(sy, sx)
        return theano.function([sx], dydx)

    def recalculate_fn(self, rval, ival):
        sy = self.svars[id(rval)]
        sx = self.svars[id(ival)]
        return theano.function([sx], sy)
