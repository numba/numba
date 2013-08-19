# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import itertools
import os
import sys
import types
import unittest
try:
    from nose.tools import nottest
except ImportError:
    def nottest(fn):
        def _nottest(*args, **kws):
            raise Exception("nose not available")
        return _nottest

import numba
from numba import *
from numba.testing import doctest_support

if numba.PY3:
    import io
else:
    import StringIO as io

jit_ = jit

if numba.PY3:
    import re
    def rewrite_doc(doc):
        doc = re.sub(r'(\d+)L', r'\1', doc)
        doc = re.sub(r'([^\.])NumbaError', r'\1numba.error.NumbaError', doc)
        doc = re.sub(r'([^\.])InvalidTemplateError', r'\1numba.error.InvalidTemplateError', doc)
        doc = re.sub(r'([^\.])UnpromotableTypeError', r'\1numba.error.UnpromotableTypeError', doc)
        return doc
    def autojit_py3doc(*args, **kwargs):
        if kwargs:
            def _inner(fun):
                fun.__doc__ = rewrite_doc(fun.__doc__)
                return autojit(*args, **kwargs)(fun)
            return _inner
        else:
            fun = args[0]
            fun.__doc__ = rewrite_doc(fun.__doc__)
            return autojit(fun)
else:
    def rewrite_doc(doc):
        return doc
    autojit_py3doc = autojit

class ASTTestCase(unittest.TestCase):
    jit = staticmethod(lambda *args, **kw: jit_(*args, **dict(kw, backend='ast')))
    backend = 'ast'
    autojit = staticmethod(autojit(backend=backend))

#------------------------------------------------------------------------
# Support for unittest in < py2.7
#------------------------------------------------------------------------

have_unit_skip = sys.version_info[:2] > (2, 6)

if have_unit_skip:
    from unittest import SkipTest
else:
    class SkipTest(Exception):
        "Skip a test in < py27"

@nottest
def skip_test(reason):
    if have_unit_skip:
        raise SkipTest(reason)
    else:
        print("Skipping: " + reason, file=sys.stderr)

def skip_if(should_skip, message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if should_skip:
                skip_test(message)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def skip_unless(should_skip, message):
    return skip_if(not should_skip, message)

def skip(message):
    return skip_if(True, message)

def checkSkipFlag(reason):
    def _checkSkipFlag(fn):
        @nottest
        def _checkSkipWrapper(self, *args, **kws):
            skip_test(reason)
        return _checkSkipWrapper
    return _checkSkipFlag

#------------------------------------------------------------------------
# Test running
#------------------------------------------------------------------------

def main():
    import sys, logging
    if '-d' in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)
        sys.argv.remove('-d')
    if '-D' in sys.argv:
        logging.getLogger().setLevel(logging.NOTSET)
        sys.argv.remove('-D')
    unittest.main()

class StdoutReplacer(object):
    def __enter__(self, *args):
        self.out = sys.stdout
        sys.stdout = io.StringIO()
        return sys.stdout

    def __exit__(self, *args):
        sys.stdout = self.out


def fix_module_doctest_py3(module):
    """
    Rewrite docs for python 3
    """
    if not numba.PY3:
        return
    if module.__doc__:
        try:
            module.__doc__ = rewrite_doc(module.__doc__)
        except:
            pass
    for name in dir(module):
        if name.startswith('__'):
            continue
        value = getattr(module, name)
        try:
            value.__doc__ = rewrite_doc(value.__doc__)
        except:
            pass

def testmod(module=None, run=True, optionflags=None,):
    """
    Tests a doctest modules with numba functions. When run in nosetests, only
    populates module.__test__, when run as main, runs the doctests.
    """
    if module is None:
        mod_globals = sys._getframe(1).f_globals
        modname = mod_globals['__name__']
        module = __import__(modname)
        # module = types.ModuleType(modname)
        # vars(module).update(mod_globals)

    fix_module_doctest_py3(module)
    doctest_support.testmod(module, run_doctests=run)

#------------------------------------------------------------------------
# Test Parametrization
#------------------------------------------------------------------------

def parametrize(*parameters, **named_parameters):
    """
    @parametrize('foo', 'bar')
    def test_func(foo_or_bar):
        print foo_or_bar # prints 'foo' or 'bar'

    or

    @parametrize(x=['foo', 'bar'], y=['baz', 'quux'])
    def test_func(x, y):
        print x, y # prints all combinations

    Generates a unittest TestCase in the function's global scope named
    'test_func_testcase' with parametrized test methods.

    ':return: The original function
    """
    if parameters and named_parameters:
        raise TypeError('Cannot specify both parameters and named_parameters')

    def decorator(func):
        class TestCase(unittest.TestCase):
            pass

        TestCase.__name__ = func.__name__
        TestCase.__module__ = func.__module__
        names = named_parameters.keys()
        values = parameters or itertools.product(*named_parameters.values())

        for i, parameter in enumerate(values):
            name = 'test_%s_%d' % (func.__name__, i)

            if names:
                def testfunc(self, parameter=parameter):
                    return func(**dict(zip(names, parameter)))
            else:
                def testfunc(self, parameter=parameter):
                    return func(parameter)

            testfunc.__name__ = name
            if func.__doc__:
                testfunc.__doc__ = func.__doc__.replace(func.__name__, name)

            setattr(TestCase, name, testfunc)

        func.__globals__[func.__name__ + '_testcase'] = TestCase
        return func

    return decorator

