from numba.testing import SerialSuite

def load_tests(loader, tests, pattern):
    suite = loader.discover("numba.hsa.tests")
    return SerialSuite(suite)
