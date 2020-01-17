from __future__ import print_function, division, absolute_import

import atexit
import collections
import functools
import io
import itertools
import os
import threading
import timeit
import math
import sys
import traceback
import weakref
from types import ModuleType
import numpy as np

from .six import *
from .errors import UnsupportedError
try:
    # preferred over pure-python StringIO due to threadsafety
    # note: parallel write to StringIO could cause data to go missing
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from numba.config import PYVERSION, MACHINE_BITS, DEVELOPER_MODE


IS_PY3 = PYVERSION >= (3, 0)

if IS_PY3:
    import builtins
    INT_TYPES = (int,)
    longint = int
    get_ident = threading.get_ident
    intern = sys.intern
    file_replace = os.replace
    asbyteint = int

    def erase_traceback(exc_value):
        """
        Erase the traceback and hanging locals from the given exception instance.
        """
        if exc_value.__traceback__ is not None:
            traceback.clear_frames(exc_value.__traceback__)
        return exc_value.with_traceback(None)

else:
    import thread
    import __builtin__ as builtins
    INT_TYPES = (int, long)
    longint = long
    get_ident = thread.get_ident
    intern = intern
    def asbyteint(x):
        # convert 1-char str into int
        return ord(x)

    if sys.platform == 'win32':
        def file_replace(src, dest):
            # Best-effort emulation of os.replace()
            try:
                os.rename(src, dest)
            except OSError:
                os.unlink(dest)
                os.rename(src, dest)
    else:
        file_replace = os.rename

    def erase_traceback(exc_value):
        """
        Erase the traceback and hanging locals from the given exception instance.
        """
        return exc_value

try:
    from inspect import signature as pysignature
    from inspect import Signature as pySignature
    from inspect import Parameter as pyParameter
except ImportError:
    try:
        from funcsigs import signature as _pysignature
        from funcsigs import Signature as pySignature
        from funcsigs import Parameter as pyParameter
        from funcsigs import BoundArguments

        def pysignature(*args, **kwargs):
            try:
                return _pysignature(*args, **kwargs)
            except ValueError as e:
                msg = ("Cannot obtain a signature for: %s. The error message "
                       "from funcsigs was: '%s'." % (args,  e.message))
                raise UnsupportedError(msg)

        # monkey patch `apply_defaults` onto `BoundArguments` cf inspect in py3
        # This patch is from https://github.com/aliles/funcsigs/pull/30/files
        # with minor modifications, and thanks!
        # See LICENSES.third-party.
        def apply_defaults(self):
            arguments = self.arguments

            # Creating a new one and not modifying in-place for thread safety.
            new_arguments = []

            for name, param in self._signature.parameters.items():
                try:
                    new_arguments.append((name, arguments[name]))
                except KeyError:
                    if param.default is not param.empty:
                        val = param.default
                    elif param.kind is _VAR_POSITIONAL:
                        val = ()
                    elif param.kind is _VAR_KEYWORD:
                        val = {}
                    else:
                        # BoundArguments was likely created by bind_partial
                        continue
                    new_arguments.append((name, val))

            self.arguments = collections.OrderedDict(new_arguments)
        BoundArguments.apply_defaults = apply_defaults
    except ImportError:
        raise ImportError("please install the 'funcsigs' package "
                          "('pip install funcsigs')")

try:
    from functools import singledispatch
except ImportError:
    try:
        import singledispatch
    except ImportError:
        raise ImportError("please install the 'singledispatch' package "
                          "('pip install singledispatch')")
    else:
        # Hotfix for https://bitbucket.org/ambv/singledispatch/issues/8/inconsistent-hierarchy-with-enum
        def _c3_merge(sequences):
            """Merges MROs in *sequences* to a single MRO using the C3 algorithm.

            Adapted from http://www.python.org/download/releases/2.3/mro/.

            """
            result = []
            while True:
                sequences = [s for s in sequences if s]   # purge empty sequences
                if not sequences:
                    return result
                for s1 in sequences:   # find merge candidates among seq heads
                    candidate = s1[0]
                    for s2 in sequences:
                        if candidate in s2[1:]:
                            candidate = None
                            break      # reject the current head, it appears later
                    else:
                        break
                if candidate is None:
                    raise RuntimeError("Inconsistent hierarchy")
                result.append(candidate)
                # remove the chosen candidate
                for seq in sequences:
                    if seq[0] == candidate:
                        del seq[0]

        singledispatch._c3_merge = _c3_merge

        singledispatch = singledispatch.singledispatch


# Mapping between operator module functions and the corresponding built-in
# operators.

BINOPS_TO_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '//': operator.floordiv,
    '/': operator.truediv,
    '%': operator.mod,
    '**': operator.pow,
    '&': operator.and_,
    '|': operator.or_,
    '^': operator.xor,
    '<<': operator.lshift,
    '>>': operator.rshift,
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
    'is': operator.is_,
    'is not': operator.is_not,
    # This one has its args reversed!
    'in': operator.contains
}

INPLACE_BINOPS_TO_OPERATORS = {
    '+=': operator.iadd,
    '-=': operator.isub,
    '*=': operator.imul,
    '//=': operator.ifloordiv,
    '/=': operator.itruediv,
    '%=': operator.imod,
    '**=': operator.ipow,
    '&=': operator.iand,
    '|=': operator.ior,
    '^=': operator.ixor,
    '<<=': operator.ilshift,
    '>>=': operator.irshift,
}

UNARY_BUITINS_TO_OPERATORS = {
    '+': operator.pos,
    '-': operator.neg,
    '~': operator.invert,
    'not': operator.not_,
    'is_true': operator.truth
}

OPERATORS_TO_BUILTINS = {
    operator.add: '+',
    operator.iadd: '+=',
    operator.sub: '-',
    operator.isub: '-=',
    operator.mul: '*',
    operator.imul: '*=',
    operator.floordiv: '//',
    operator.ifloordiv: '//=',
    operator.truediv: '/',
    operator.itruediv: '/=',
    operator.mod: '%',
    operator.imod: '%=',
    operator.pow: '**',
    operator.ipow: '**=',
    operator.and_: '&',
    operator.iand: '&=',
    operator.or_: '|',
    operator.ior: '|=',
    operator.xor: '^',
    operator.ixor: '^=',
    operator.lshift: '<<',
    operator.ilshift: '<<=',
    operator.rshift: '>>',
    operator.irshift: '>>=',
    operator.eq: '==',
    operator.ne: '!=',
    operator.lt: '<',
    operator.le: '<=',
    operator.gt: '>',
    operator.ge: '>=',
    operator.is_: 'is',
    operator.is_not: 'is not',
    # This one has its args reversed!
    operator.contains: 'in',
    # Unary
    operator.pos: '+',
    operator.neg: '-',
    operator.invert: '~',
    operator.not_: 'not',
    operator.truth: 'is_true',
}

HAS_MATMUL_OPERATOR = sys.version_info >= (3, 5)

if not IS_PY3:
    BINOPS_TO_OPERATORS['/?'] = operator.div
    INPLACE_BINOPS_TO_OPERATORS['/?='] = operator.idiv
    OPERATORS_TO_BUILTINS[operator.div] = '/?'
    OPERATORS_TO_BUILTINS[operator.idiv] = '/?'
if HAS_MATMUL_OPERATOR:
    BINOPS_TO_OPERATORS['@'] = operator.matmul
    INPLACE_BINOPS_TO_OPERATORS['@='] = operator.imatmul


_shutting_down = False

def _at_shutdown():
    global _shutting_down
    _shutting_down = True


def shutting_down(globals=globals):
    """
    Whether the interpreter is currently shutting down.
    For use in finalizers, __del__ methods, and similar; it is advised
    to early bind this function rather than look it up when calling it,
    since at shutdown module globals may be cleared.
    """
    # At shutdown, the attribute may have been cleared or set to None.
    v = globals().get('_shutting_down')
    return v is True or v is None


# weakref.finalize registers an exit function that runs all finalizers for
# which atexit is True. Some of these finalizers may call shutting_down() to
# check whether the interpreter is shutting down. For this to behave correctly,
# we need to make sure that _at_shutdown is called before the finalizer exit
# function. Since atexit operates as a LIFO stack, we first contruct a dummy
# finalizer then register atexit to ensure this ordering.
weakref.finalize(lambda: None, lambda: None)
atexit.register(_at_shutdown)


class ConfigOptions(object):
    OPTIONS = {}

    def __init__(self):
        self._values = self.OPTIONS.copy()

    def set(self, name, value=True):
        if name not in self.OPTIONS:
            raise NameError("Invalid flag: %s" % name)
        self._values[name] = value

    def unset(self, name):
        self.set(name, False)

    def _check_attr(self, name):
        if name not in self.OPTIONS:
            raise AttributeError("Invalid flag: %s" % name)

    def __getattr__(self, name):
        self._check_attr(name)
        return self._values[name]

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(ConfigOptions, self).__setattr__(name, value)
        else:
            self._check_attr(name)
            self._values[name] = value

    def __repr__(self):
        return "Flags(%s)" % ', '.join('%s=%s' % (k, v)
                                       for k, v in self._values.items()
                                       if v is not False)

    def copy(self):
        copy = type(self)()
        copy._values = self._values.copy()
        return copy

    def __eq__(self, other):
        return isinstance(other, ConfigOptions) and other._values == self._values

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(sorted(self._values.items())))


class SortedMap(Mapping):
    """Immutable
    """

    def __init__(self, seq):
        self._values = []
        self._index = {}
        for i, (k, v) in enumerate(sorted(seq)):
            self._index[k] = i
            self._values.append((k, v))

    def __getitem__(self, k):
        i = self._index[k]
        return self._values[i][1]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(k for k, v in self._values)


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise AssertionError("key already in dictionary: %r" % (key,))
        super(UniqueDict, self).__setitem__(key, value)


# Django's cached_property
# see https://docs.djangoproject.com/en/dev/ref/utils/#django.utils.functional.cached_property

class cached_property(object):
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.

    Optional ``name`` argument allows you to make cached properties of other
    methods. (e.g.  url = cached_property(get_absolute_url, name='url') )
    """
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


def runonce(fn):
    @functools.wraps(fn)
    def inner():
        if not inner._ran:
            res = fn()
            inner._result = res
            inner._ran = True
        return inner._result

    inner._ran = False
    return inner


def bit_length(intval):
    """
    Return the number of bits necessary to represent integer `intval`.
    """
    assert isinstance(intval, INT_TYPES)
    if intval >= 0:
        return len(bin(intval)) - 2
    else:
        return len(bin(-intval - 1)) - 2


def stream_list(lst):
    """
    Given a list, return an infinite iterator of iterators.
    Each iterator iterates over the list from the last seen point up to
    the current end-of-list.

    In effect, each iterator will give the newly appended elements from the
    previous iterator instantiation time.
    """
    def sublist_iterator(start, stop):
        return iter(lst[start:stop])

    start = 0
    while True:
        stop = len(lst)
        yield sublist_iterator(start, stop)
        start = stop


class BenchmarkResult(object):
    def __init__(self, func, records, loop):
        self.func = func
        self.loop = loop
        self.records = np.array(records) / loop
        self.best = np.min(self.records)

    def __repr__(self):
        name = getattr(self.func, "__name__", self.func)
        args = (name, self.loop, self.records.size, format_time(self.best))
        return "%20s: %10d loops, best of %d: %s per loop" % args


def format_time(tm):
    units = "s ms us ns ps".split()
    base = 1
    for unit in units[:-1]:
        if tm >= base:
            break
        base /= 1000
    else:
        unit = units[-1]
    return "%.1f%s" % (tm / base, unit)


def benchmark(func, maxsec=1):
    timer = timeit.Timer(func)
    number = 1
    result = timer.repeat(1, number)
    # Too fast to be measured
    while min(result) / number == 0:
        number *= 10
        result = timer.repeat(3, number)
    best = min(result) / number
    if best >= maxsec:
        return BenchmarkResult(func, result, number)
        # Scale it up to make it close the maximum time
    max_per_run_time = maxsec / 3 / number
    number = max(max_per_run_time / best / 3, 1)
    # Round to the next power of 10
    number = int(10 ** math.ceil(math.log10(number)))
    records = timer.repeat(3, number)
    return BenchmarkResult(func, records, number)


RANGE_ITER_OBJECTS = (builtins.range,)
if PYVERSION < (3, 0):
    RANGE_ITER_OBJECTS += (builtins.xrange,)
    try:
        from future.types.newrange import newrange
        RANGE_ITER_OBJECTS += (newrange,)
    except ImportError:
        pass


def logger_hasHandlers(logger):
    # Backport from python3.5 logging implementation of `.hasHandlers()`
    c = logger
    rv = False
    while c:
        if c.handlers:
            rv = True
            break
        if not c.propagate:
            break
        else:
            c = c.parent
    return rv


# A dummy module for dynamically-generated functions
_dynamic_modname = '<dynamic>'
_dynamic_module = ModuleType(_dynamic_modname)
_dynamic_module.__builtins__ = moves.builtins


def chain_exception(new_exc, old_exc):
    """Set the __cause__ attribute on *new_exc* for explicit exception
    chaining.  Returns the inplace modified *new_exc*.
    """
    if DEVELOPER_MODE:
        new_exc.__cause__ = old_exc
    return new_exc
