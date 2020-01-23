from numba.testing import SerialSuite
from numba.testing import load_testsuite
from os.path import dirname, join
from numba.dppy.dppy_driver import driver as ocldrv

def load_tests(loader, tests, pattern):

    suite = SerialSuite()
    this_dir = dirname(__file__)

    if ocldrv.has_gpu_device() or ocldrv.has_cpu_device():
	suite.addTests(load_testsuite(loader, join(this_dir, 'dppy')))
    else:
        print("skipped DPPY tests")

    return suite
