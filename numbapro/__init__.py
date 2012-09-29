# as _internal contains the license checking code, we import it here
from . import _internal


def test(verbosity=2, failfast=False):
    import unittest
    import pkgutil
    import sys

    test_package_names = ['numbapro.tests.basic_vectorize',
                          'numbapro.tests.llvm_cbuilder_tests',
                          'numbapro.tests.parallel_vectorize',
                          'numbapro.tests.stream_vectorize']

    loader = unittest.TestLoader()

    # Find all test scripts in test packages
    test_module_names = []
    for name in test_package_names:
        test_module_names.extend([
                name + '.' + module
                for _,module,_ in pkgutil.iter_modules(
                           __import__(name, fromlist=['']).__path__)])

    suite = loader.loadTestsFromNames(test_module_names)
    # The default stream doesn't work in Windows IPython qtconsole
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=True, stream=sys.stdout)
    return runner.run(suite)

def drop_in_gdb(addr=None, type='int'):
    import os, signal
    if addr is not None:
        print 'watch *(%s *) %s' % (type, hex(addr))
    os.kill(os.getpid(), signal.SIGINT)

import numbapro.array_expressions
