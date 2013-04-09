"""
Adpated from http://wiki.cython.org/FAQ#HowcanIrundoctestsinCythoncode.28pyxfiles.29.3F
Use the testmod function from test_support, don't use this directly.
====================================================================

Cython-compatible wrapper for doctest.testmod().

Usage example, assuming a Cython module mymod.pyx is compiled.
This is run from the command line, passing a command to Python:
python -c "import cydoctest, mymod; cydoctest.testmod(mymod)"

(This still won't let a Cython module run its own doctests
when called with "python mymod.py", but it's pretty close.
Further options can be passed to testmod() as desired, e.g.
verbose=True.)
"""
import sys
import unittest
import doctest
import inspect

import numba.decorators
from numba import numbawrapper

doctest_options = doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE


def from_module(module, object):
    """
    Return true if the given object is defined in the given module.
    """
    if module is None:
        return True
    elif inspect.getmodule(object) is not None:
        return module is inspect.getmodule(object)
    elif inspect.isfunction(object):
        return module.__dict__ is object.__globals__
    elif inspect.isclass(object):
        return module.__name__ == object.__module__
    elif hasattr(object, '__module__'):
        return module.__name__ == object.__module__
    elif isinstance(object, property):
        return True # [XX] no way not be sure.
    else:
        raise ValueError("object must be a class or function")

def fix_module_doctest(module):
    """
    Extract docstrings from cython functions, that would be skipped by doctest
    otherwise.
    """
    module.__test__ = {}
    for name in dir(module):
        value = getattr(module, name)
        if (isinstance(value, numbawrapper.NumbaWrapper) and
                from_module(module, value.py_func) and value.py_func.__doc__):
            module.__test__[name] = value.py_func.__doc__
        elif (inspect.isbuiltin(value) and isinstance(value.__doc__, str) and
                  from_module(module, value)):
            module.__test__[name] = value.__doc__

class MyDocTestFinder(doctest.DocTestFinder):
    def find(self, obj, **kws):
        res = doctest.DocTestFinder.find(self, obj, **kws)
        return res

def testmod(m=None, run_doctests=True, optionflags=doctest_options,
            verbosity=2):
    """
    Fix a Cython module's doctests, then call doctest.testmod()
    """
    fix_module_doctest(m)
    if run_doctests:
        finder = MyDocTestFinder(exclude_empty=False)
        suite = doctest.DocTestSuite(m, test_finder=finder,
                                     optionflags=optionflags)
        result =  unittest.TextTestRunner(verbosity=verbosity).run(suite)
        if not result.wasSuccessful():
            raise Exception("Doctests failed: %s" % result)
