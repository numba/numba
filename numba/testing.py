from __future__ import print_function, division, absolute_import
import unittest


def test():
    loader = unittest.TestLoader()
    startdir = "numba.tests"
    suite = loader.discover(startdir)

    runner = unittest.TextTestRunner(descriptions=True, verbosity=1,
                                     buffer=True)
    result = runner.run(suite)
    return result.wasSuccessful()

