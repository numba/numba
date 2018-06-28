from __future__ import print_function, division, absolute_import

from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import sys
import warnings
from numba import config, errors

from . import utils
from .errors import (NotDefinedError, RedefinedError, VerificationError,
                     ConstantInferenceError)

# terminal color markup
_termcolor = errors.termcolor()

class Loc(object):
    """Source location

    """

    def __init__(self, filename, line, col=None):
        self.filename = filename
        self.line = line
        self.col = col

    @classmethod
    def from_function_id(cls, func_id):
        return cls(func_id.filename, func_id.firstlineno)

    def __repr__(self):
        return "Loc(filename=%s, line=%s, col=%s)" % (self.filename,
                                                      self.line, self.col)

    def __str__(self):
        if self.col is not None:
            return "%s (%s:%s)" % (self.filename, self.line, self.col)
        else:
            return "%s (%s)" % (self.filename, self.line)

    def strformat(self, nlines_up=2):
        try:
            # Try to get a relative path
            # ipython/jupyter input just returns as self.filename
            path = os.path.relpath(self.filename)
        except ValueError:
            # Fallback to absolute path if error occurred in getting the
            # relative path.
            # This may happen on windows if the drive is different
            path = os.path.abspath(self.filename)

        lines = linecache.getlines(path)

        ret = [] # accumulates output
        if lines and self.line:

            def count_spaces(string):
                spaces = 0
                for x in itertools.takewhile(str.isspace, str(string)):
                    spaces += 1
                return spaces

            selected = lines[self.line - nlines_up:self.line]
            # see if selected contains a definition
            def_found = False
            for x in selected:
                if 'def ' in x:
                    def_found = True

            # no definition found, try and find one
            if not def_found:
                # try and find a def, go backwards from error line
                fn_name = None
                for x in reversed(lines[:self.line - 1]):
                    if 'def ' in x:
                        fn_name = x
                        break
                if fn_name:
                    ret.append(fn_name)
                    spaces = count_spaces(x)
                    ret.append(' '*(4 + spaces) + '<source elided>\n')

            ret.extend(selected[:-1])
            ret.append(_termcolor.highlight(selected[-1]))

            # point at the problem with a caret
            spaces = count_spaces(selected[-1])
            ret.append(' '*(spaces) + _termcolor.indicate("^"))

        # if in the REPL source may not be available
        if not ret:
            ret = "<source missing, REPL in use?>"

        err = _termcolor.filename('\nFile "%s", line %d:')+'\n%s'
        tmp = err % (path, self.line, _termcolor.code(''.join(ret)))
        return tmp

    def with_lineno(self, line, col=None):
        """
        Return a new Loc with this line number.
        """
        return type(self)(self.filename, line, col)


class VarMap(object):
    def __init__(self):
        self._con = {}

    def define(self, name, var):
        if name in self._con:
            raise RedefinedError(name)
        else:
            self._con[name] = var

    def get(self, name):
        try:
            return self._con[name]
        except KeyError:
            raise NotDefinedError(name)

    def __contains__(self, name):
        return name in self._con

    def __len__(self):
        return len(self._con)

    def __repr__(self):
        return pprint.pformat(self._con)

    def __hash__(self):
        return hash(self.name)

    def __iter__(self):
        return self._con.iterkeys()


class Inst(object):
    """
    Base class for all IR instructions.
    """

    def list_vars(self):
        """
        List the variables used (read or written) by the instruction.
        """
        raise NotImplementedError

    def _rec_list_vars(self, val):
        """
        A recursive helper used to implement list_vars() in subclasses.
        """
        if isinstance(val, Var):
            return [val]
        elif isinstance(val, Inst):
            return val.list_vars()
        elif isinstance(val, (list, tuple)):
            lst = []
            for v in val:
                lst.extend(self._rec_list_vars(v))
            return lst
        elif isinstance(val, dict):
            lst = []
            for v in val.values():
                lst.extend(self._rec_list_vars(v))
            return lst
        else:
            return []


class Stmt(Inst):
    """
    Base class for IR statements (instructions which can appear on their
    own in a Block).
    """
    # Whether this statement ends its basic block (i.e. it will either jump
    # to another block or exit the function).
    is_terminator = False
    # Whether this statement exits the function.
    is_exit = False

    def list_vars(self):
        return self._rec_list_vars(self.__dict__)


class Terminator(Stmt):
    """
    IR statements that are terminators: the last statement in a block.
    A terminator must either:
    - exit the function
    - jump to a block

    All subclass of Terminator must override `.get_targets()` to return a list
    of jump targets.
    """
    is_terminator = True

    def get_targets(self):
        raise NotImplementedError(type(self))


class Expr(Inst):
    """
    An IR expression (an instruction which can only be part of a larger
    statement).
    """

    def __init__(self, op, loc, **kws):
        self.op = op
        self.loc = loc
        self._kws = kws

    def __getattr__(self, name):
        if name.startswith('_'):
            return Inst.__getattr__(self, name)
        return self._kws[name]

    def __setattr__(self, name, value):
        if name in ('op', 'loc', '_kws'):
            self.__dict__[name] = value
        else:
            self._kws[name] = value

    @classmethod
    def binop(cls, fn, lhs, rhs, loc):
        op = 'binop'
        return cls(op=op, loc=loc, fn=fn, lhs=lhs, rhs=rhs,
                   static_lhs=UNDEFINED, static_rhs=UNDEFINED)

    @classmethod
    def inplace_binop(cls, fn, immutable_fn, lhs, rhs, loc):
        op = 'inplace_binop'
        return cls(op=op, loc=loc, fn=fn, immutable_fn=immutable_fn,
                   lhs=lhs, rhs=rhs,
                   static_lhs=UNDEFINED, static_rhs=UNDEFINED)

    @classmethod
    def unary(cls, fn, value, loc):
        op = 'unary'
        return cls(op=op, loc=loc, fn=fn, value=value)

    @classmethod
    def call(cls, func, args, kws, loc, vararg=None):
        op = 'call'
        return cls(op=op, loc=loc, func=func, args=args, kws=kws,
                   vararg=vararg)

    @classmethod
    def build_tuple(cls, items, loc):
        op = 'build_tuple'
        return cls(op=op, loc=loc, items=items)

    @classmethod
    def build_list(cls, items, loc):
        op = 'build_list'
        return cls(op=op, loc=loc, items=items)

    @classmethod
    def build_set(cls, items, loc):
        op = 'build_set'
        return cls(op=op, loc=loc, items=items)

    @classmethod
    def build_map(cls, items, size, loc):
        op = 'build_map'
        return cls(op=op, loc=loc, items=items, size=size)

    @classmethod
    def pair_first(cls, value, loc):
        op = 'pair_first'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def pair_second(cls, value, loc):
        op = 'pair_second'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def getiter(cls, value, loc):
        op = 'getiter'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def iternext(cls, value, loc):
        op = 'iternext'
        return cls(op=op, loc=loc, value=value)

    @classmethod
    def exhaust_iter(cls, value, count, loc):
        op = 'exhaust_iter'
        return cls(op=op, loc=loc, value=value, count=count)

    @classmethod
    def getattr(cls, value, attr, loc):
        op = 'getattr'
        return cls(op=op, loc=loc, value=value, attr=attr)

    @classmethod
    def getitem(cls, value, index, loc):
        op = 'getitem'
        return cls(op=op, loc=loc, value=value, index=index)

    @classmethod
    def static_getitem(cls, value, index, index_var, loc):
        op = 'static_getitem'
        return cls(op=op, loc=loc, value=value, index=index,
                   index_var=index_var)

    @classmethod
    def cast(cls, value, loc):
        """
        A node for implicit casting at the return statement
        """
        op = 'cast'
        return cls(op=op, value=value, loc=loc)

    @classmethod
    def make_function(cls, name, code, closure, defaults, loc):
        """
        A node for making a function object.
        """
        op = 'make_function'
        return cls(op=op, name=name, code=code, closure=closure, defaults=defaults, loc=loc)

    def __repr__(self):
        if self.op == 'call':
            args = ', '.join(str(a) for a in self.args)
            pres_order = self._kws.items() if config.DIFF_IR == 0 else sorted(self._kws.items())
            kws = ', '.join('%s=%s' % (k, v) for k, v in pres_order)
            vararg = '*%s' % (self.vararg,) if self.vararg is not None else ''
            arglist = ', '.join(filter(None, [args, vararg, kws]))
            return 'call %s(%s)' % (self.func, arglist)
        elif self.op == 'binop':
            return '%s %s %s' % (self.lhs, self.fn, self.rhs)
        else:
            pres_order = self._kws.items() if config.DIFF_IR == 0 else sorted(self._kws.items())
            args = ('%s=%s' % (k, v) for k, v in pres_order)
            return '%s(%s)' % (self.op, ', '.join(args))

    def list_vars(self):
        return self._rec_list_vars(self._kws)

    def infer_constant(self):
        raise ConstantInferenceError('%s' % self, loc=self.loc)


class SetItem(Stmt):
    """
    target[index] = value
    """

    def __init__(self, target, index, value, loc):
        self.target = target
        self.index = index
        self.value = value
        self.loc = loc

    def __repr__(self):
        return '%s[%s] = %s' % (self.target, self.index, self.value)


class StaticSetItem(Stmt):
    """
    target[constant index] = value
    """

    def __init__(self, target, index, index_var, value, loc):
        self.target = target
        self.index = index
        self.index_var = index_var
        self.value = value
        self.loc = loc

    def __repr__(self):
        return '%s[%r] = %s' % (self.target, self.index, self.value)


class DelItem(Stmt):
    """
    del target[index]
    """

    def __init__(self, target, index, loc):
        self.target = target
        self.index = index
        self.loc = loc

    def __repr__(self):
        return 'del %s[%s]' % (self.target, self.index)


class SetAttr(Stmt):
    def __init__(self, target, attr, value, loc):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc

    def __repr__(self):
        return '(%s).%s = %s' % (self.target, self.attr, self.value)


class DelAttr(Stmt):
    def __init__(self, target, attr, loc):
        self.target = target
        self.attr = attr
        self.loc = loc

    def __repr__(self):
        return 'del (%s).%s' % (self.target, self.attr)


class StoreMap(Stmt):
    def __init__(self, dct, key, value, loc):
        self.dct = dct
        self.key = key
        self.value = value
        self.loc = loc

    def __repr__(self):
        return '%s[%s] = %s' % (self.dct, self.key, self.value)


class Del(Stmt):
    def __init__(self, value, loc):
        self.value = value
        self.loc = loc

    def __str__(self):
        return "del %s" % self.value


class Raise(Terminator):
    is_exit = True

    def __init__(self, exception, loc):
        self.exception = exception
        self.loc = loc

    def __str__(self):
        return "raise %s" % self.exception

    def get_targets(self):
        return []


class StaticRaise(Terminator):
    """
    Raise an exception class and arguments known at compile-time.
    Note that if *exc_class* is None, a bare "raise" statement is implied
    (i.e. re-raise the current exception).
    """
    is_exit = True

    def __init__(self, exc_class, exc_args, loc):
        self.exc_class = exc_class
        self.exc_args = exc_args
        self.loc = loc

    def __str__(self):
        if self.exc_class is None:
            return "raise"
        elif self.exc_args is None:
            return "raise %s" % (self.exc_class,)
        else:
            return "raise %s(%s)" % (self.exc_class,
                                     ", ".join(map(repr, self.exc_args)))

    def get_targets(self):
        return []


class Return(Terminator):
    """
    Return to caller.
    """
    is_exit = True

    def __init__(self, value, loc):
        self.value = value
        self.loc = loc

    def __str__(self):
        return 'return %s' % self.value

    def get_targets(self):
        return []


class Jump(Terminator):
    """
    Unconditional branch.
    """

    def __init__(self, target, loc):
        self.target = target
        self.loc = loc

    def __str__(self):
        return 'jump %s' % self.target

    def get_targets(self):
        return [self.target]


class Branch(Terminator):
    """
    Conditional branch.
    """

    def __init__(self, cond, truebr, falsebr, loc):
        self.cond = cond
        self.truebr = truebr
        self.falsebr = falsebr
        self.loc = loc

    def __str__(self):
        return 'branch %s, %s, %s' % (self.cond, self.truebr, self.falsebr)

    def get_targets(self):
        return [self.truebr, self.falsebr]


class Assign(Stmt):
    """
    Assign to a variable.
    """
    def __init__(self, value, target, loc):
        self.value = value
        self.target = target
        self.loc = loc

    def __str__(self):
        return '%s = %s' % (self.target, self.value)


class Print(Stmt):
    """
    Print some values.
    """
    def __init__(self, args, vararg, loc):
        self.args = args
        self.vararg = vararg
        # Constant-inferred arguments
        self.consts = {}
        self.loc = loc

    def __str__(self):
        return 'print(%s)' % ', '.join(str(v) for v in self.args)


class Yield(Inst):
    def __init__(self, value, loc, index):
        self.value = value
        self.loc = loc
        self.index = index

    def __str__(self):
        return 'yield %s' % (self.value,)

    def list_vars(self):
        return [self.value]


class Arg(object):
    def __init__(self, name, index, loc):
        self.name = name
        self.index = index
        self.loc = loc

    def __repr__(self):
        return 'arg(%d, name=%s)' % (self.index, self.name)

    def infer_constant(self):
        raise ConstantInferenceError('%s' % self, loc=self.loc)


class Const(object):
    def __init__(self, value, loc):
        self.value = value
        self.loc = loc

    def __repr__(self):
        return 'const(%s, %s)' % (type(self.value).__name__, self.value)

    def infer_constant(self):
        return self.value

class Global(object):
    def __init__(self, name, value, loc):
        self.name = name
        self.value = value
        self.loc = loc

    def __str__(self):
        return 'global(%s: %s)' % (self.name, self.value)

    def infer_constant(self):
        return self.value

    def __deepcopy__(self, memo):
        # don't copy value since it can fail (e.g. modules)
        # value is readonly and doesn't need copying
        return Global(self.name, self.value, copy.deepcopy(self.loc))


class FreeVar(object):
    """
    A freevar, as loaded by LOAD_DECREF.
    (i.e. a variable defined in an enclosing non-global scope)
    """

    def __init__(self, index, name, value, loc):
        # index inside __code__.co_freevars
        self.index = index
        # variable name
        self.name = name
        # frozen value
        self.value = value
        self.loc = loc

    def __str__(self):
        return 'freevar(%s: %s)' % (self.name, self.value)

    def infer_constant(self):
        return self.value


class Var(object):
    """
    Attributes
    -----------
    - scope: Scope

    - name: str

    - loc: Loc
        Definition location
    """

    def __init__(self, scope, name, loc):
        self.scope = scope
        self.name = name
        self.loc = loc

    def __repr__(self):
        return 'Var(%s, %s)' % (self.name, self.loc)

    def __str__(self):
        return self.name

    @property
    def is_temp(self):
        return self.name.startswith("$")


class Intrinsic(object):
    """
    A low-level "intrinsic" function.  Suitable as the callable of a "call"
    expression.

    The given *name* is backend-defined and will be inserted as-is
    in the generated low-level IR.
    The *type* is the equivalent Numba signature of calling the intrinsic.
    """

    def __init__(self, name, type, args):
        self.name = name
        self.type = type
        self.loc = None
        self.args = args

    def __repr__(self):
        return 'Intrinsic(%s, %s, %s)' % (self.name, self.type, self.loc)

    def __str__(self):
        return self.name


class Scope(object):
    """
    Attributes
    -----------
    - parent: Scope
        Parent scope

    - localvars: VarMap
        Scope-local variable map

    - loc: Loc
        Start of scope location

    """

    def __init__(self, parent, loc):
        self.parent = parent
        self.localvars = VarMap()
        self.loc = loc
        self.redefined = defaultdict(int)

    def define(self, name, loc):
        """
        Define a variable
        """
        v = Var(scope=self, name=name, loc=loc)
        self.localvars.define(v.name, v)
        return v

    def get(self, name):
        """
        Refer to a variable.  Returns the latest version.
        """
        if name in self.redefined:
            name = "%s.%d" % (name, self.redefined[name])
        return self.get_exact(name)

    def get_exact(self, name):
        """
        Refer to a variable.  The returned variable has the exact
        name (exact variable version).
        """
        try:
            return self.localvars.get(name)
        except NotDefinedError:
            if self.has_parent:
                return self.parent.get(name)
            else:
                raise

    def get_or_define(self, name, loc):
        if name in self.redefined:
            name = "%s.%d" % (name, self.redefined[name])

        v = Var(scope=self, name=name, loc=loc)
        if name not in self.localvars:
            return self.define(name, loc)
        else:
            return self.localvars.get(name)

    def redefine(self, name, loc, rename=True):
        """
        Redefine if the name is already defined
        """
        if name not in self.localvars:
            return self.define(name, loc)
        elif not rename:
            # Must use the same name if the variable is a cellvar, which
            # means it could be captured in a closure.
            return self.localvars.get(name)
        else:
            ct = self.redefined[name]
            self.redefined[name] = ct + 1
            newname = "%s.%d" % (name, ct + 1)
            return self.define(newname, loc)

    def make_temp(self, loc):
        n = len(self.localvars)
        v = Var(scope=self, name='$%d' % n, loc=loc)
        self.localvars.define(v.name, v)
        return v

    @property
    def has_parent(self):
        return self.parent is not None

    def __repr__(self):
        return "Scope(has_parent=%r, num_vars=%d, %s)" % (self.has_parent,
                                                          len(self.localvars),
                                                          self.loc)


class Block(object):
    """A code block

    """

    def __init__(self, scope, loc):
        self.scope = scope
        self.body = []
        self.loc = loc

    def copy(self):
        block = Block(self.scope, self.loc)
        block.body = self.body[:]
        return block

    def find_exprs(self, op=None):
        """
        Iterate over exprs of the given *op* in this block.
        """
        for inst in self.body:
            if isinstance(inst, Assign):
                expr = inst.value
                if isinstance(expr, Expr):
                    if op is None or expr.op == op:
                        yield expr

    def find_insts(self, cls=None):
        """
        Iterate over insts of the given class in this block.
        """
        for inst in self.body:
            if isinstance(inst, cls):
                yield inst

    def find_variable_assignment(self, name):
        """
        Returns the assignment inst associated with variable "name", None if
        it cannot be found.
        """
        for x in self.find_insts(cls=Assign):
            if x.target.name == name:
                return x
        return None

    def prepend(self, inst):
        assert isinstance(inst, Stmt)
        self.body.insert(0, inst)

    def append(self, inst):
        assert isinstance(inst, Stmt)
        self.body.append(inst)

    def remove(self, inst):
        assert isinstance(inst, Stmt)
        del self.body[self.body.index(inst)]

    def clear(self):
        del self.body[:]

    def dump(self, file=None):
        # Avoid early bind of sys.stdout as default value
        file = file or sys.stdout
        for inst in self.body:
            if hasattr(inst, 'dump'):
                inst.dump(file)
            else:
                inst_vars = sorted(str(v) for v in inst.list_vars())
                print('    %-40s %s' % (inst, inst_vars), file=file)

    @property
    def terminator(self):
        return self.body[-1]

    @property
    def is_terminated(self):
        return self.body and self.body[-1].is_terminator

    def verify(self):
        if not self.is_terminated:
            raise VerificationError("Missing block terminator")
            # Only the last instruction can be a terminator
        for inst in self.body[:-1]:
            if inst.is_terminator:
                raise VerificationError("Terminator before the last "
                                        "instruction")

    def insert_after(self, stmt, other):
        """
        Insert *stmt* after *other*.
        """
        index = self.body.index(other)
        self.body.insert(index + 1, stmt)

    def insert_before_terminator(self, stmt):
        assert isinstance(stmt, Stmt)
        assert self.is_terminated
        self.body.insert(-1, stmt)

    def __repr__(self):
        return "<ir.Block at %s>" % (self.loc,)


class Loop(object):
    __slots__ = "entry", "exit"

    def __init__(self, entry, exit):
        self.entry = entry
        self.exit = exit

    def __repr__(self):
        args = self.entry, self.exit
        return "Loop(entry=%s, exit=%s)" % args


class FunctionIR(object):

    def __init__(self, blocks, is_generator, func_id, loc,
                 definitions, arg_count, arg_names):
        self.blocks = blocks
        self.is_generator = is_generator
        self.func_id = func_id
        self.loc = loc
        self.arg_count = arg_count
        self.arg_names = arg_names

        self._definitions = definitions

        self._reset_analysis_variables()

    def _reset_analysis_variables(self):
        from . import consts

        self._consts = consts.ConstantInference(self)

        # Will be computed by PostProcessor
        self.generator_info = None
        self.variable_lifetime = None
        # { ir.Block: { variable names (potentially) alive at start of block } }
        self.block_entry_vars = {}

    def derive(self, blocks, arg_count=None, arg_names=None,
               force_non_generator=False):
        """
        Derive a new function IR from this one, using the given blocks,
        and possibly modifying the argument count and generator flag.

        Post-processing will have to be run again on the new IR.
        """
        firstblock = blocks[min(blocks)]

        new_ir = copy.copy(self)
        new_ir.blocks = blocks
        new_ir.loc = firstblock.loc
        if force_non_generator:
            new_ir.is_generator = False
        if arg_count is not None:
            new_ir.arg_count = arg_count
        if arg_names is not None:
            new_ir.arg_names = arg_names
        new_ir._reset_analysis_variables()
        # Make fresh func_id
        new_ir.func_id = new_ir.func_id.derive()
        return new_ir

    def copy(self):
        new_ir = copy.copy(self)
        blocks = {}
        block_entry_vars = {}
        for label, block in self.blocks.items():
            new_block = block.copy()
            blocks[label] = new_block
            if block in self.block_entry_vars:
                block_entry_vars[new_block] = self.block_entry_vars[block]
        new_ir.blocks = blocks
        new_ir.block_entry_vars = block_entry_vars
        return new_ir

    def get_block_entry_vars(self, block):
        """
        Return a set of variable names possibly alive at the beginning of
        the block.
        """
        return self.block_entry_vars[block]

    def infer_constant(self, name):
        """
        Try to infer the constant value of a given variable.
        """
        if isinstance(name, Var):
            name = name.name
        return self._consts.infer_constant(name)

    def get_definition(self, value, lhs_only=False):
        """
        Get the definition site for the given variable name or instance.
        A Expr instance is returned by default, but if lhs_only is set
        to True, the left-hand-side variable is returned instead.
        """
        lhs = value
        while True:
            if isinstance(value, Var):
                lhs = value
                name = value.name
            elif isinstance(value, str):
                lhs = value
                name = value
            else:
                return lhs if lhs_only else value
            defs = self._definitions[name]
            if len(defs) == 0:
                raise KeyError("no definition for %r"
                               % (name,))
            if len(defs) > 1:
                raise KeyError("more than one definition for %r"
                               % (name,))
            value = defs[0]

    def dump(self, file=None):
        # Avoid early bind of sys.stdout as default value
        file = file or sys.stdout
        for offset, block in sorted(self.blocks.items()):
            print('label %s:' % (offset,), file=file)
            block.dump(file=file)

    def dump_generator_info(self, file=None):
        file = file or sys.stdout
        gi = self.generator_info
        print("generator state variables:", sorted(gi.state_vars), file=file)
        for index, yp in sorted(gi.yield_points.items()):
            print("yield point #%d: live variables = %s, weak live variables = %s"
                  % (index, sorted(yp.live_vars), sorted(yp.weak_live_vars)),
                  file=file)


# A stub for undefined global reference
UNDEFINED = object()
