from __future__ import print_function, division, absolute_import
import numba.unittest_support as unittest


def test():
    loader = unittest.TestLoader()
    startdir = "numba.tests"
    suite = loader.discover(startdir)

    runner = unittest.TextTestRunner(descriptions=True, verbosity=2,
                                     buffer=True)
    result = runner.run(suite)
    return result.wasSuccessful()

