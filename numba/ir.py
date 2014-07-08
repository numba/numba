from __future__ import print_function, division, absolute_import
import sys
import os
import pprint
from collections import defaultdict


class RedefinedError(NameError):
    pass


class NotDefinedError(NameError):
    pass


class VerificationError(Exception):
    pass


class Loc(object):
    """Source location

    """

    def __init__(self, filename, line, col=None):
        self.filename = filename
        self.line = line
        self.col = col

    def __repr__(self):
        return "Loc(filename=%s, line=%s, col=%s)" % (self.filename,
                                                      self.line, self.col)

    def __str__(self):
        if self.col is not None:
            return "%s (%s:%s)" % (self.filename, self.line, self.col)
        else:
            return "%s (%s)" % (self.filename, self.line)

    def strformat(self):
        try:
            # Try to get a relative path
            path = os.path.relpath(self.filename)
        except ValueError:
            # Fallback to absolute path if error occured in getting the
            # relative path.
            # This may happen on windows if the drive is different
            path = os.path.abspath(self.filename)
        return 'File "%s", line %d' % (path, self.line)


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


class Stmt(object):
    is_terminator = False


class Expr(object):
    def __init__(self, op, loc, **kws):
        self.op = op
        self.loc = loc
        self._kws = kws
        for k, v in kws.items():
            setattr(self, k, v)

    @classmethod
    def binop(cls, fn, lhs, rhs, loc):
        op = 'binop'
        return cls(op=op, loc=loc, fn=fn, lhs=lhs, rhs=rhs)

    @classmethod
    def unary(cls, fn, value, loc):
        op = 'unary'
        return cls(op=op, loc=loc, fn=fn, value=value)

    @classmethod
    def call(cls, func, args, kws, loc):
        op = 'call'
        return cls(op=op, loc=loc, func=func, args=args, kws=kws)

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
    def build_map(cls, size, loc):
        op = 'build_map'
        return cls(op=op, loc=loc, size=size)

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
    def static_getitem(cls, value, index, loc):
        op = 'static_getitem'
        return cls(op=op, loc=loc, value=value, index=index)

    def __repr__(self):
        if self.op == 'call':
            args = ', '.join(str(a) for a in self.args)
            kws = ', '.join('%s=%s' % (k, v) for k, v in self.kws)
            return 'call %s(%s, %s)' % (self.func, args, kws)
        elif self.op == 'binop':
            return '%s %s %s' % (self.lhs, self.fn, self.rhs)
        else:
            args = ('%s=%s' % (k, v) for k, v in self._kws.items())
            return '%s(%s)' % (self.op, ', '.join(args))

    def list_vars(self):
        for v in self._kws.values():
            if isinstance(v, Var):
                return v


class SetItem(Stmt):
    def __init__(self, target, index, value, loc):
        self.target = target
        self.index = index
        self.value = value
        self.loc = loc

    def __repr__(self):
        return '%s[%s] = %s' % (self.target, self.index, self.value)


class SetAttr(Stmt):
    def __init__(self, target, attr, value, loc):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc

    def __repr__(self):
        return '(%s).%s = %s' % (self.target, self.attr, self.value)


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


class Raise(Stmt):
    is_terminator = True

    def __init__(self, exception, loc):
        self.exception = exception
        self.loc = loc

    def __str__(self):
        return "raise %s" % self.exception


class Return(Stmt):
    is_terminator = True

    def __init__(self, value, loc):
        self.value = value
        self.loc = loc

    def __str__(self):
        return 'return %s' % self.value


class Jump(Stmt):
    is_terminator = True

    def __init__(self, target, loc):
        self.target = target
        self.loc = loc

    def __str__(self):
        return 'jump %s' % self.target


class Branch(Stmt):
    is_terminator = True

    def __init__(self, cond, truebr, falsebr, loc):
        self.cond = cond
        self.truebr = truebr
        self.falsebr = falsebr
        self.loc = loc

    def __str__(self):
        return 'branch %s, %s, %s' % (self.cond, self.truebr, self.falsebr)


class Assign(Stmt):
    def __init__(self, value, target, loc):
        self.value = value
        self.target = target
        self.target.predecessor = value # set predecessor
        self.loc = loc

    def __str__(self):
        return '%s = %s' % (self.target, self.value)


class Const(object):
    def __init__(self, value, loc):
        self.value = value
        self.loc = loc

    def __repr__(self):
        return 'const(%s, %s)' % (type(self.value), self.value)


class Global(object):
    def __init__(self, name, value, loc):
        self.name = name
        self.value = value
        self.loc = loc

    def __str__(self):
        return 'global(%s: %s)' % (self.name, self.value)


class Var(object):
    """
    Attributes
    -----------
    - scope: Scope

    - name: str

    - loc: Loc
        Definition location

    - predecessor:
        predecessor of the value
    """

    def __init__(self, scope, name, loc):
        self.scope = scope
        self.name = name
        self.loc = loc
        self.predecessor = None

    def __repr__(self):
        return 'Var(%s, %s)' % (self.name, self.loc)

    def __str__(self):
        return self.name

    @property
    def is_temp(self):
        return self.name.startswith("$")


class Intrinsic(object):
    """
    For inserting intrinsic node into the IR
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
        Refer to a variable
        """
        if name in self.redefined:
            name = "%s.%d" % (name, self.redefined[name])
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

    def redefine(self, name, loc):
        """
        Redefine if the name is already defined
        """
        if name not in self.localvars:
            return self.define(name, loc)
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

    def append(self, inst):
        assert isinstance(inst, Stmt)
        self.body.append(inst)

    def remove(self, inst):
        assert isinstance(inst, Stmt)
        del self.body[self.body.index(inst)]

    def dump(self, file=sys.stdout):
        for inst in self.body:
            print('  ', inst, file=file)

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

    def insert_before_terminator(self, stmt):
        assert isinstance(stmt, Stmt)
        assert self.is_terminated
        self.body.insert(-1, stmt)


class Loop(object):
    __slots__ = "entry", "condition", "body", "exit"

    def __init__(self, entry, exit, condition=None):
        self.entry = entry
        self.condition = condition
        self.body = []
        self.exit = exit

    def valid(self):
        try:
            self.verify()
        except VerificationError:
            return False
        else:
            return True

    def verify(self):
        if self.entry is None:
            raise VerificationError("Missing entry block")
        if self.condition is None:
            raise VerificationError("Missing condition block")
        if self.exit is None:
            raise VerificationError("Missing exit block")
        if not self.body:
            raise VerificationError("Missing body block")

    def __repr__(self):
        args = self.entry, self.condition, self.body, self.exit
        return "Loop(entry=%s, condition=%s, body=%s, exit=%s)" % args


# A stub for undefined global reference
UNDEFINED = object()
