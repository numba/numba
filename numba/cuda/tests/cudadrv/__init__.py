from numba.tests import SerialSuite

def load_tests(loader, tests, pattern):
    suite = loader.discover("numba.cuda.tests.cudadrv")
    return SerialSuite(suite)
