from __future__ import print_function

import weakref

import numba.unittest_support as unittest
from numba.controlflow import CFGraph, Loop
from numba.compiler import compile_isolated, Flags
from numba import types
from .support import TestCase

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")

forceobj_flags = Flags()
forceobj_flags.set("force_pyobject")

no_pyobj_flags = Flags()


class _Dummy(object):

    def __init__(self, recorder, name):
        self.recorder = recorder
        self.name = name
        recorder.add_dummy(self)

    def __add__(self, other):
        assert isinstance(other, _Dummy)
        return _Dummy(self.recorder, "%s + %s" % (self.name, other.name))

    def __iadd__(self, other):
        assert isinstance(other, _Dummy)
        return _Dummy(self.recorder, "%s += %s" % (self.name, other.name))

    def __iter__(self):
        return _DummyIterator(self.recorder, "iter(%s)" % self.name)


class _DummyIterator(_Dummy):

    count = 0

    def __next__(self):
        if self.count >= 3:
            raise StopIteration
        self.count += 1
        return _Dummy(self.recorder, "%s#%s" % (self.name, self.count))

    next = __next__


class RefRecorder(object):

    def __init__(self):
        self._events = []
        self._wrs = {}

    def make_dummy(self, name):
        return _Dummy(self, name)

    def add_dummy(self, dummy):
        wr = weakref.ref(dummy, self._on_disposal)
        self._wrs[wr] = dummy.name

    __call__ = make_dummy

    def _on_disposal(self, wr):
        name = self._wrs.pop(wr)
        self._events.append(name)

    @property
    def alive(self):
        return [wr() for wr in self._wrs]
        for wr in list(self._wrs):
            o = wr()
            if o is not None:
                objs.append(o)
        return objs

    @property
    def recorded(self):
        return self._events


def simple_usecase1(rec):
    a = rec('a')
    b = rec('b')
    c = rec('c')
    a = b + c
    d = a + a   # b + c + b + c
    return d

def simple_usecase2(rec):
    a = rec('a')
    b = rec('b')
    x = a
    y = x
    a = None
    return y

def looping_usecase1(rec):
    a = rec('a')
    b = rec('b')
    c = rec('c')
    x = b
    for y in a:
        x = x + y
    x = x + c
    return x


class TestObjLifetime(TestCase):
    """
    Test lifetime of Python objects inside jit-compiled functions.
    """

    def compile(self, pyfunc):
        cr = compile_isolated(pyfunc, (), flags=forceobj_flags)
        self.__cres = cr
        return cr.entry_point

    def compile_and_record(self, pyfunc):
        rec = RefRecorder()
        cr = compile_isolated(pyfunc, (), flags=forceobj_flags)
        cr.entry_point(rec)
        return rec

    def assertRecordOrder(self, rec, expected):
        """
        Check that the *expected* markers occur in that order in *rec*'s
        recorded events.
        """
        actual = []
        recorded = rec.recorded
        for d in recorded:
            if d in expected:
                actual.append(d)
        self.assertEqual(actual, expected, recorded)

    def test_simple1(self):
        rec = self.compile_and_record(simple_usecase1)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', 'b', 'b + c', 'b + c + b + c'])
        self.assertRecordOrder(rec, ['a', 'c', 'b + c', 'b + c + b + c'])

    def test_simple2(self):
        rec = self.compile_and_record(simple_usecase2)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['b', 'a'])

    def test_looping1(self):
        rec = self.compile_and_record(looping_usecase1)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', 'b', 'c'])


if __name__ == "__main__":
    unittest.main()
