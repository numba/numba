from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType

from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION

opcode_info = namedtuple('opcode_info', ['argsize'])

# The following offset is used as a hack to inject a NOP at the start of the
# bytecode. So that function starting with `while True` will not have block-0
# as a jump target. The Lowerer puts argument initialization at block-0.
_FIXED_OFFSET = 2


def get_function_object(obj):
    """
    Objects that wraps function should provide a "__numba__" magic attribute
    that contains a name of an attribute that contains the actual python
    function object.
    """
    attr = getattr(obj, "__numba__", None)
    if attr:
        return getattr(obj, attr)
    return obj


def get_code_object(obj):
    "Shamelessly borrowed from llpython"
    return getattr(obj, '__code__', getattr(obj, 'func_code', None))


def _as_opcodes(seq):
    lst = []
    for s in seq:
        c = dis.opmap.get(s)
        if c is not None:
            lst.append(c)
    return lst


JREL_OPS = frozenset(dis.hasjrel)
JABS_OPS = frozenset(dis.hasjabs)
JUMP_OPS = JREL_OPS | JABS_OPS
TERM_OPS = frozenset(_as_opcodes(['RETURN_VALUE', 'RAISE_VARARGS']))
EXTENDED_ARG = dis.EXTENDED_ARG
HAVE_ARGUMENT = dis.HAVE_ARGUMENT


class ByteCodeInst(object):
    '''
    Attributes
    ----------
    - offset:
        byte offset of opcode
    - opcode:
        opcode integer value
    - arg:
        instruction arg
    - lineno:
        -1 means unknown
    '''
    __slots__ = 'offset', 'next', 'opcode', 'opname', 'arg', 'lineno'

    def __init__(self, offset, opcode, arg, nextoffset):
        self.offset = offset
        self.next = nextoffset
        self.opcode = opcode
        self.opname = dis.opname[opcode]
        self.arg = arg
        self.lineno = -1  # unknown line number

    @property
    def is_jump(self):
        return self.opcode in JUMP_OPS

    @property
    def is_terminator(self):
        return self.opcode in TERM_OPS

    def get_jump_target(self):
        # With Python 3.10 the addressing of "bytecode" instructions has
        # changed from using bytes to using 16-bit words instead. As a
        # consequence the code to determine where a jump will lead had to be
        # adapted.
        # See also:
        # https://bugs.python.org/issue26647
        # https://bugs.python.org/issue27129
        # https://github.com/python/cpython/pull/25069
        assert self.is_jump
        if PYVERSION == (3, 11):
            if self.opcode in (dis.opmap[k]
                               for k in ("JUMP_BACKWARD",
                                         "POP_JUMP_BACKWARD_IF_TRUE",
                                         "POP_JUMP_BACKWARD_IF_FALSE",
                                         "POP_JUMP_BACKWARD_IF_NONE",
                                         "POP_JUMP_BACKWARD_IF_NOT_NONE",)):
                return self.offset - (self.arg - 1) * 2
        elif PYVERSION > (3, 11):
            raise NotImplementedError(PYVERSION)

        if PYVERSION >= (3, 10):
            if self.opcode in JREL_OPS:
                return self.next + self.arg * 2
            else:
                assert self.opcode in JABS_OPS
                return self.arg * 2 - 2
        else:
            if self.opcode in JREL_OPS:
                return self.next + self.arg
            else:
                assert self.opcode in JABS_OPS
                return self.arg

    def __repr__(self):
        return '%s(arg=%s, lineno=%d)' % (self.opname, self.arg, self.lineno)

    @property
    def block_effect(self):
        """Effect of the block stack
        Returns +1 (push), 0 (none) or -1 (pop)
        """
        if self.opname.startswith('SETUP_'):
            return 1
        elif self.opname == 'POP_BLOCK':
            return -1
        else:
            return 0


CODE_LEN = 1
ARG_LEN = 1
NO_ARG_LEN = 1

OPCODE_NOP = dis.opname.index('NOP')


# Adapted from Lib/dis.py
def _unpack_opargs(code):
    """
    Returns a 4-int-tuple of
    (bytecode offset, opcode, argument, offset of next bytecode).
    """
    extended_arg = 0
    n = len(code)
    offset = i = 0
    while i < n:
        op = code[i]
        i += CODE_LEN
        if op >= HAVE_ARGUMENT:
            arg = code[i] | extended_arg
            for j in range(ARG_LEN):
                arg |= code[i + j] << (8 * j)
            i += ARG_LEN
            if op == EXTENDED_ARG:
                # This is a deviation from what dis does...
                # In python 3.11 it seems like EXTENDED_ARGs appear more often
                # and are also used as jump targets. So as to not have to do
                # "book keeping" for where EXTENDED_ARGs have been "skipped"
                # they are replaced with NOPs so as to provide a legal jump
                # target and also ensure that the bytecode offsets are correct.
                yield (offset, OPCODE_NOP, arg, i)
                extended_arg = arg << 8 * ARG_LEN
                offset = i
                continue
        else:
            arg = None
            i += NO_ARG_LEN

        extended_arg = 0
        yield (offset, op, arg, i)
        offset = i  # Mark inst offset at first extended


def _patched_opargs(bc_stream):
    """Patch the bytecode stream.

    - Adds a NOP bytecode at the start to avoid jump target being at the entry.
    """
    # Injected NOP
    yield (0, OPCODE_NOP, None, _FIXED_OFFSET)
    # Adjust bytecode offset for the rest of the stream
    for offset, opcode, arg, nextoffset in bc_stream:
        # If the opcode has an absolute jump target, adjust it.
        if opcode in JABS_OPS:
            arg += _FIXED_OFFSET
        yield offset + _FIXED_OFFSET, opcode, arg, nextoffset + _FIXED_OFFSET


class ByteCodeIter(object):
    def __init__(self, code):
        self.code = code
        self.iter = iter(_patched_opargs(_unpack_opargs(self.code.co_code)))

    def __iter__(self):
        return self

    def _fetch_opcode(self):
        return next(self.iter)

    def next(self):
        offset, opcode, arg, nextoffset = self._fetch_opcode()
        return offset, ByteCodeInst(offset=offset, opcode=opcode, arg=arg,
                                    nextoffset=nextoffset)

    __next__ = next

    def read_arg(self, size):
        buf = 0
        for i in range(size):
            _offset, byte = next(self.iter)
            buf |= byte << (8 * i)
        return buf


class _ByteCode(object):
    """
    The decoded bytecode of a function, and related information.
    """
    __slots__ = ('func_id', 'co_names', 'co_varnames', 'co_consts',
                 'co_cellvars', 'co_freevars', 'exception_entries',
                 'table', 'labels')

    def __init__(self, func_id):
        code = func_id.code

        labels = set(x + _FIXED_OFFSET for x in dis.findlabels(code.co_code))
        labels.add(0)

        # A map of {offset: ByteCodeInst}
        table = OrderedDict(ByteCodeIter(code))
        self._compute_lineno(table, code)

        self.func_id = func_id
        self.co_names = code.co_names
        self.co_varnames = code.co_varnames
        self.co_consts = code.co_consts
        self.co_cellvars = code.co_cellvars
        self.co_freevars = code.co_freevars

        self.table = table
        self.labels = sorted(labels)

    @classmethod
    def _compute_lineno(cls, table, code):
        """
        Compute the line numbers for all bytecode instructions.
        """
        for offset, lineno in dis.findlinestarts(code):
            adj_offset = offset + _FIXED_OFFSET
            if adj_offset in table:
                table[adj_offset].lineno = lineno
        # Assign unfilled lineno
        # Start with first bytecode's lineno
        known = code.co_firstlineno
        for inst in table.values():
            if inst.lineno >= 0:
                known = inst.lineno
            else:
                inst.lineno = known
        return table

    def __iter__(self):
        return iter(self.table.values())

    def __getitem__(self, offset):
        return self.table[offset]

    def __contains__(self, offset):
        return offset in self.table

    def dump(self):
        def label_marker(i):
            if i[1].offset in self.labels:
                return '>'
            else:
                return ' '

        return '\n'.join('%s %10s\t%s' % ((label_marker(i),) + i)
                         for i in self.table.items()
                         if i[1].opname != "CACHE")

    @classmethod
    def _compute_used_globals(cls, func, table, co_consts, co_names):
        """
        Compute the globals used by the function with the given
        bytecode table.
        """
        d = {}
        globs = func.__globals__
        builtins = globs.get('__builtins__', utils.builtins)
        if isinstance(builtins, ModuleType):
            builtins = builtins.__dict__
        # Look for LOAD_GLOBALs in the bytecode
        for inst in table.values():
            if inst.opname == 'LOAD_GLOBAL':
                name = co_names[_fix_LOAD_GLOBAL_arg(inst.arg)]
                if name not in d:
                    try:
                        value = globs[name]
                    except KeyError:
                        value = builtins[name]
                    d[name] = value
        # Add globals used by any nested code object
        for co in co_consts:
            if isinstance(co, CodeType):
                subtable = OrderedDict(ByteCodeIter(co))
                d.update(cls._compute_used_globals(func, subtable,
                                                   co.co_consts, co.co_names))
        return d

    def get_used_globals(self):
        """
        Get a {name: value} map of the globals used by this code
        object and any nested code objects.
        """
        return self._compute_used_globals(self.func_id.func, self.table,
                                          self.co_consts, self.co_names)


def _fix_LOAD_GLOBAL_arg(arg):
    if PYVERSION >= (3, 11):
        assert PYVERSION == (3, 11) # reminder to check newer versions
        return arg >> 1
    return arg


class ByteCodePy311(_ByteCode):
    def __init__(self, func_id):
        super().__init__(func_id)

        def fixup_eh(ent):
            from dis import _ExceptionTableEntry
            # Patch up the exception table offset
            # because we add a NOP in _patched_opargs
            out = _ExceptionTableEntry(
                start=ent.start + _FIXED_OFFSET, end=ent.end + _FIXED_OFFSET,
                target=ent.target + _FIXED_OFFSET,
                depth=ent.depth, lasti=ent.lasti,
            )
            return out

        entries = dis.Bytecode(func_id.code).exception_entries
        self.exception_entries = tuple(map(fixup_eh, entries))

    def find_exception_entry(self, offset):
        """
        Returns the exception entry for the given instruction offset
        """
        candidates = []
        for ent in self.exception_entries:
            if ent.start <= offset <= ent.end:
                candidates.append((ent.depth, ent))
        if candidates:
            ent = max(candidates)[1]
            return ent


if PYVERSION == (3, 11):
    ByteCode = ByteCodePy311
elif PYVERSION < (3, 11):
    ByteCode = _ByteCode
else:
    raise NotImplementedError(PYVERSION)


class FunctionIdentity(serialize.ReduceMixin):
    """
    A function's identity and metadata.

    Note this typically represents a function whose bytecode is
    being compiled, not necessarily the top-level user function
    (the two might be distinct, e.g. in the `@generated_jit` case).
    """
    _unique_ids = itertools.count(1)

    @classmethod
    def from_function(cls, pyfunc):
        """
        Create the FunctionIdentity of the given function.
        """
        func = get_function_object(pyfunc)
        code = get_code_object(func)
        pysig = utils.pysignature(func)
        if not code:
            raise errors.ByteCodeSupportError(
                "%s does not provide its bytecode" % func)

        try:
            func_qualname = func.__qualname__
        except AttributeError:
            func_qualname = func.__name__

        self = cls()
        self.func = func
        self.func_qualname = func_qualname
        self.func_name = func_qualname.split('.')[-1]
        self.code = code
        self.module = inspect.getmodule(func)
        self.modname = (utils._dynamic_modname
                        if self.module is None
                        else self.module.__name__)
        self.is_generator = inspect.isgeneratorfunction(func)
        self.pysig = pysig
        self.filename = code.co_filename
        self.firstlineno = code.co_firstlineno
        self.arg_count = len(pysig.parameters)
        self.arg_names = list(pysig.parameters)

        # Even the same function definition can be compiled into
        # several different function objects with distinct closure
        # variables, so we make sure to disambiguate using an unique id.
        uid = next(cls._unique_ids)
        self.unique_name = '{}${}'.format(self.func_qualname, uid)
        self.unique_id = uid

        return self

    def derive(self):
        """Copy the object and increment the unique counter.
        """
        return self.from_function(self.func)

    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
        return dict(pyfunc=self.func)

    @classmethod
    def _rebuild(cls, pyfunc):
        """
        NOTE: part of ReduceMixin protocol
        """
        return cls.from_function(pyfunc)
