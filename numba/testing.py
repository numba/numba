from __future__ import print_function, division, absolute_import
import unittest
import sys


def test():
    loader = unittest.TestLoader()
    startdir = "numba.tests"
    suite = loader.discover(startdir)
    result = unittest.TextTestResult(stream=sys.stderr, descriptions=True,
                                     verbosity=1)
    suite.run(result)
    if result.wasSuccessful():
        return True
    else:
        print(result)
        return False

