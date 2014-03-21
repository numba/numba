from __future__ import print_function, division, absolute_import


def discover_tests(startdir):
    import numba.unittest_support as unittest

    loader = unittest.TestLoader()
    suite = loader.discover(startdir)
    return suite


def run_tests(suite, descriptions=True, verbosity=2, buffer=True,
              failfast=True):
    import numba.unittest_support as unittest

    runner = unittest.TextTestRunner(descriptions=descriptions,
                                     verbosity=verbosity,
                                     buffer=buffer, failfast=failfast)
    result = runner.run(suite)
    return result


def test():
    suite = discover_tests("numba.tests")
    return run_tests(suite).wasSuccessful()


def multitest():
    """
    Run tests in multiple processes.
    """
    import numba.unittest_support as unittest
    import multiprocessing as mp

    loader = unittest.TestLoader()
    startdir = "numba.tests"
    suites = loader.discover(startdir)

    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(_multiruntest, suites)
    return all(results)


def _multiruntest(suite):
    import numba.unittest_support as unittest

    runner = unittest.TextTestRunner(descriptions=False, verbosity=0,
                                     buffer=True)
    result = runner.run(suite)
    return result.wasSuccessful()
