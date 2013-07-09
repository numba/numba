import inspect
import dis
from collections import defaultdict

from .errors import error_context
from .bytecode import ByteCode

class SymbolicExecution(object):
    def __init__(self, func):
        self.func = func
        self.bytecode = ByteCode(func)
        self.runtime = SymbolicRuntime(self, self.bytecode)

        # ignore any POP_TOP at the first of a block for python26
        self._ignore_first_pop_top = set()

    def visit(self):
        # prepare arguments
        argspec = inspect.getargspec(self.func)
        assert not argspec.defaults, "does not support defaults"
        assert not argspec.varargs, "does not support varargs"
        assert not argspec.keywords, "does not support keywords"
        for argnum, argname in enumerate(argspec.args):
            self.runtime.load_argument(num=argnum, name=argname)
        # visit every bytecode
        for inst in self.bytecode:
            with error_context(inst.lineno):
                oldblock = self.runtime.curblock
                self.runtime.on_next_inst(inst)
                if (oldblock is not self.runtime.curblock and
                        inst.opname == 'POP_TOP' and
                        self.runtime.curblock in self._ignore_first_pop_top):
                    continue    # skip
                if self.runtime.curblock.terminator is not None:
                    continue    # skip
                attr = 'visit_' + inst.opname
                func = getattr(self, attr, self.generic_visit)
                func(inst)

    def generic_visit(self, inst):
        raise NotImplementedError(inst)

    def visit_LOAD_FAST(self, inst):
        name = self.runtime.varnames[inst.arg]
        self.runtime.load_name(name)

    def visit_STORE_FAST(self, inst):
        name = self.runtime.varnames[inst.arg]
        self.runtime.assign(name, self.runtime.pop())

    def visit_BINARY_ADD(self, inst):
        self.runtime.binary_op('+')

    def visit_BINARY_SUBTRACT(self, inst):
        self.runtime.binary_op('-')

    def visit_BINARY_MULTIPLY(self, inst):
        self.runtime.binary_op('*')

    def visit_BINARY_DIVIDE(self, inst):
        self.runtime.binary_op('//')

    def visit_BINARY_FLOOR_DIVIDE(self, inst):
        self.runtime.binary_op('//')

    def visit_BINARY_TRUE_DIVIDE(self, inst):
        self.runtime.binary_op('/')

    def visit_BINARY_MODULO(self, inst):
        self.runtime.binary_op('%')

    def visit_BINARY_POWER(self, inst):
        self.runtime.binary_op('**')

    def visit_COMPARE_OP(self, inst):
        self.runtime.binary_op(dis.cmp_op[inst.arg])

    def visit_POP_JUMP_IF_TRUE(self, inst):
        falsebr = inst.next
        truebr = inst.arg
        self.runtime.jump_if(truebr, falsebr)

    def visit_POP_JUMP_IF_FALSE(self, inst):
        truebr = inst.next
        falsebr = inst.arg
        self.runtime.jump_if(truebr, falsebr)

    def visit_JUMP_IF_TRUE(self, inst):
        falsebr = inst.next
        truebr = inst.arg
        self._ignore_first_pop_top.add(self.runtime.blocks[truebr])
        self._ignore_first_pop_top.add(self.runtime.blocks[falsebr])
        self.runtime.jump_if(truebr, falsebr)

    def visit_JUMP_IF_FALSE(self, inst):
        truebr = inst.next
        falsebr = inst.arg
        self._ignore_first_pop_top.add(self.runtime.blocks[truebr])
        self._ignore_first_pop_top.add(self.runtime.blocks[falsebr])
        self.runtime.jump_if(truebr, falsebr)

    def visit_RETURN_VALUE(self, inst):
        self.runtime.ret()

#---------------------------------------------------------------------------
# Internals

class SymbolicRuntime(object):
    def __init__(self, parent, bytecode):
        self.parent = parent

        self.varnames = bytecode.code.co_varnames
        self.consts = bytecode.code.co_consts
        self.names = bytecode.code.co_names

        self.stack = []
        self.blocks = dict((offset, Block(offset))
                           for offset in bytecode.labels)
        self.curblock = self.blocks[0]

    def on_next_inst(self, inst):
        oldblock = self.curblock
        self.curblock = self.blocks.get(inst.offset, oldblock)

    def push(self, value):
        "push value onto the stack"
        self.stack.append(value)

    def pop(self):
        "pop value from the stack"
        return self.stack.pop()

    def peek(self):
        "peek at the TOS"
        return elf.stack[-1]

    def insert(self, inst):
        inst.block = self.curblock
        self.curblock.code.append(inst)

    def terminate(self, inst):
        assert self.curblock.terminator is None
        self.curblock.terminator = inst

    def assign(self, name, value):
        self.curblock.vars[name] = value

    def load_argument(self, num, name):
        inst = Inst('arg', argnum=num)
        ref = self.insert(inst)
        self.assign(name, inst)

    def load_name(self, name):
        self.push(self.curblock.vars[name])
        raise Exception("TODO: variable loading from preivous block")

    def call(self, func, *args, **kws):
        callinst = Inst('call', callee=func, args=args, kws=kws)
        self.push(callinst)

    def binary_op(self, op):
        rhs = self.pop()
        lhs = self.pop()
        self.call('+', lhs, rhs)

    def jump_if(self, truebr, falsebr):
        cond = self.pop()
        self.terminate(Inst('bra', cond=cond, truebr=truebr, falsebr=falsebr))

    def ret(self):
        val = self.pop()
        if val.opcode == 'const' and val.value is None:
            self.terminate(Inst('retvoid'))
        else:
            self.terminate(Inst('ret', value=val))

class Block(object):
    def __init__(self, offset):
        self.offset = offset
        self.code = []
        self.vars = ValDefTable()
        self.terminator = None


class ValDefTable(object):
    def __init__(self):
        self._map = defaultdict(list)

    def __setitem__(self, name, value):
        self._map[name].append(value)

    def __getitem__(self, name):
        try:
            return self._map[name][-1]
        except IndexError:
            raise KeyError(name)


class Inst(object):
    def __init__(self, opcode, **kwargs):
        self.opcode = opcode
        self.attrs = tuple(kwargs.keys())
        self.block = None
        for k, v in kwargs.items():
            assert not hasattr(self, k)
            setattr(self, k, v)


