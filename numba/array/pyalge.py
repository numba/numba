"""
PyAlge

Provides pattern matching through the "Case" class and (minimal support of)
algebraic data type (ADT).

ADT are implemented as namedtuple.

"""

from __future__ import print_function, absolute_import
from collections import namedtuple
import functools
import operator
import tokenize
import inspect


RESERVED = set(['force', 'recurse', 'otherwise'])


class MissingCaseError(ValueError):
    pass


class PatternSyntaxError(ValueError):
    pass


class PatternContextError(ValueError):
    pass


class NoMatch(object):
    """Used internally to indicate where a case does not match.
"""
    pass


def _get_code(func):
    try:
        return func.__code__
    except AttributeError:
        return func.func_code


def _prepare(cls):
    """Insert "_case_ofs" attribute to the class
"""
    if not hasattr(cls, "_case_ofs"):
        ofs = []
        for fd in dir(cls):
            if not fd.startswith('_') and fd not in RESERVED:
                fn = getattr(cls, fd)
                firstline = _get_code(fn._inner).co_firstlineno
                ofs.append((firstline, fn))
                # Order cases by lineno
        cls._case_ofs = tuple(zip(*sorted(ofs)))[1]


class _Common(object):
    def otherwise(self, value):
        """Default is to raise MissingCaseError exception with `value` as
argument.

Can be overridden.
"""
        raise MissingCaseError(value)

    def recurse(self, value):
        return type(self)(value=value, state=self.state)


class Case(_Common):
    """A case-class is used to describe a pattern matching dispatch logic.
Each pattern is described as a method with the `of` decorator.

Details
-------
The `__new__` method of Case is overridden to not return an instance of
Case. One will use a subclass of Case as a function.
"""

    def __new__(cls, value, state=None):
        """
Args
----
value:
object being matched
state:
mutable internal state accessible as `self.state`
"""
        _prepare(cls)
        obj = object.__new__(cls)
        obj.state = state
        obj.value = value
        return obj.__process()

    def __process(self):
        """The actual matching/dispatch.
Returns the result of the match.
"""
        ofs = self._case_ofs
        for case in ofs:
            res = case(self)
            if res is not NoMatch:
                # Matches
                return res

        # Run default
        return self.otherwise(self.value)


# XXX: this is experimental
class LazyCase(_Common):
    """For walking a structure lazily.

"""

    def __init__(self, value, state=None):
        _prepare(type(self))
        self.value = value
        self.state = state

    def force(self):
        stack = [self]
        while stack:
            tos = stack.pop()
            res = tos._dispatch(stack)
            if res is not NoMatch:
                return res

    def _dispatch(self, stack):
        ofs = self._case_ofs
        for case in ofs:
            res = case(self)
            if res is not NoMatch:
                # Matches
                pending = []
                for item in res:
                    if isinstance(item, LazyCase):
                        pending.append(item)
                    else:
                        return item
                stack.extend(reversed(pending))
                return NoMatch

        # Run default
        res = self.otherwise(self.value)
        pending = []
        for item in res:
            if isinstance(item, LazyCase):
                pending.append(item)
            else:
                return item
        return NoMatch


class _TypePattern(namedtuple("_TypePattern", ["typ", "body"])):
    def match(self, match, input):
        if isinstance(input, self.typ):
            if len(self.body) != len(input):
                # Insufficient fields
                return

            # Try to match the fields
            for field, fdvalue in zip(self.body, input):
                m = field.match(_Match(), fdvalue)
                if m is not None:
                    match.bindings.update(m.bindings)
                else:
                    return
            return match


class _Binding(namedtuple("Binding", ["name"])):
    def match(self, match, input):
        assert self.name not in match.bindings, "duplicated binding"
        match.bindings[self.name] = input
        return match


class _Ignored(object):
    def match(self, match, input):
        return match


class _PatternParser(object):
    """Pattern parser

Overhead of the library will almost likely be in parsing. Luckily,
we parse once at class creation.
"""

    def __init__(self, pat, env):
        self.pat = pat
        self.bindings = set()
        self.env = env

        lines = iter(self.pat.split())
        tokens = tokenize.generate_tokens(lambda: next(lines))
        self._nexttoken = lambda: next(tokens)
        self.pbtok = None
        self.result = None

    def get_token(self):
        if self.pbtok:
            token, self.pbtok = self.pbtok, None
        else:
            token = self._nexttoken()
        return token

    def putback(self, token):
        assert self.pbtok is None, "internal error: lookahead(1)"
        self.pbtok = token

    def parse(self):
        typename = self.expect_name()
        self.result = self.parse_typebody(typename)

    def peek(self):
        tok = self.get_token()
        self.putback(tok)
        return tok

    def parse_typebody(self, typename):
        if not typename[0].isupper():
            raise PatternSyntaxError("type name must start with uppercase "
                                     "letter")
        self.expect_lparen()
        body = []
        while True:
            body.append(self.expect_type_or_binding())
            if not self.is_comma():
                break
            elif self.peek()[1] == ')':
                break
        self.expect_rparen()

        try:
            typcls = self.env[typename]
        except KeyError:
            raise PatternContextError("unknown type reference: %s" % typename)
        else:
            return _TypePattern(typcls, body)

    def expect_type_or_binding(self):
        name = self.expect_name()
        if name[0].isupper():
            # is type
            return self.parse_typebody(name)
        else:
            # is binding
            return self.parse_binding(name)

    def parse_binding(self, name):
        ignored = name[0].startswith('_')
        if not name[0].islower() and not ignored:
            raise PatternSyntaxError("binding name must start with lowercase "
                                     "letter")
        if ignored:
            return _Ignored()
        else:
            return _Binding(name)

    def expect_name(self):
        token = self.get_token()
        tokty, tokstr = token[:2]
        if tokty != tokenize.NAME:
            raise PatternSyntaxError("expected name but got %r" % tokstr)
        return tokstr

    def expect_lparen(self):
        token = self.get_token()
        tokstr = token[1]
        if tokstr != '(':
            raise PatternSyntaxError("expected '('; got %r" % tokstr)

    def expect_rparen(self):
        token = self.get_token()
        tokstr = token[1]
        if tokstr != ')':
            raise PatternSyntaxError("expected ')'; got %r" % tokstr)

    def is_comma(self):
        token = self.get_token()
        tokstr = token[1]
        if tokstr != ',':
            self.putback(token)
            return False
        else:
            return True


class _Match(object):
    def __init__(self):
        self.bindings = {}


def of(pat):
    """Decorator for methods of Case to describe the pattern.

Args
----
pat: str

Patterns are like writing tuples (of tuples (of ...)) for the type
structure to match against. Names starting with a lowercase letter are
used as binding slots that the matcher will capture and used as argument
to the action function, the function being decorated. Names that starts
with a underscore '_' is ignored. Names starting with a uppercase letter
are type names.
"""

    # Get globals from caller's frame
    glbls = inspect.currentframe().f_back.f_globals
    # Parse pattern
    parser = _PatternParser(pat, glbls)
    parser.parse()
    matcher = parser.result

    def decor(fn):
        assert fn.__name__ not in RESERVED, "function name is reserved"

        @functools.wraps(fn)
        def closure(self):
            match = matcher.match(_Match(), self.value)
            if match is not None:
                self.match = match
                return fn(self, **match.bindings)
            else:
                return NoMatch

        closure._inner = fn
        return closure

    return decor


class Data(object):
    """Mutable record type.

Behaves like a namedtuple with mutable fields.
"""

    def __init__(self, *args, **kws):
        remaining = set(self._fields_)
        # Handle *args
        for k, v in zip(self._fields_, args):
            remaining.remove(k)
            setattr(self, k, v)

        # Handle **kws
        for k, v in kws.items():
            if k not in remaining:
                raise AttributeError("redefining %s" % k)
            else:
                remaining.remove(k)
                setattr(self, k, v)

        if remaining:
            raise AttributeError(
                "attribute required: %s" % ', '.join(remaining))

    def __iter__(self):
        lst = [getattr(self, f) for f in self._fields_]
        return iter(lst)

    def __getitem__(self, num):
        return getattr(self, self._fields_[num])

    def __setattr__(self, key, value):
        if key not in self._fields_:
            raise AttributeError("%s does not have attribute %s" %
                                 (type(self), key))
        super(Data, self).__setattr__(key, value)

    def __len__(self):
        return len(self._fields_)

    def __cmp__(self, other):
        diff = (cmp(a, b) for a, b in zip(self, other))
        for c in diff:
            if c != 0:
                return c
        return 0

    def __hash__(self):
        return functools.reduce(operator.add, map(hash, self))

    def __repr__(self):
        attrs = ('%s=%r' % (k, v) for k, v in zip(self._fields_, self))
        return "%s(%s)" % (self._constr_, ', '.join(attrs))

    def __str__(self):
        attrs = ('%s=%s' % (k, v) for k, v in zip(self._fields_, self))
        return "%s(%s)" % (self._constr_, ', '.join(attrs))

    def _asdict(self):
        return dict(zip(self._fields_, self))


def datatype(name, fields):
    """Constructor for product type.
Used like `collections.namedtuple`.

Args
----
name: str
Type name. Must start with a uppercase letter.
fields: sequence of str
Sequence of field names.
"""
    assert name[0].isupper(), "Type name must start with uppercase letter."
    gl = {"_fields_": fields,
          "__slots__": fields,
          "_constr_": name}
    return type(name, (Data,), gl)

