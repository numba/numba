from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

#def test(verbosity=2, failfast=False):
#    import unittest
#    import pkgutil
#    import sys
#
#    test_package_names = ['numbapro.tests.basic_vectorize',
#                          'numbapro.tests.parallel_vectorize',
#                          'numbapro.tests.stream_vectorize',
#                          'numbapro.tests.vectorize_pointer',]
#
#    loader = unittest.TestLoader()
#
#    # Find all test scripts in test packages
#    test_module_names = []
#    for name in test_package_names:
#        test_module_names.extend([
#                name + '.' + module
#                for _,module,_ in pkgutil.iter_modules(
#                           __import__(name, fromlist=['']).__path__)])
#
#    suite = loader.loadTestsFromNames(test_module_names)
#    # The default stream doesn't work in Windows IPython qtconsole
#    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=True, stream=sys.stdout)
#    return runner.run(suite)

EXCLUDE_TEST_PACKAGES = []

from numba import whitelist_match, map_returncode_to_message

def exclude_package_dirs(dirs, cuda=False):
    excludes = EXCLUDE_TEST_PACKAGES
    if not cuda:
        excludes.append('cuda')
    for exclude_pkg in excludes:
        if exclude_pkg in dirs:
            dirs.remove(exclude_pkg)

def qualified_test_name(root):
    import os
    qname = root.replace("/", ".").replace("\\", ".").replace(os.sep, ".") + "."
    offset = qname.rindex('numbapro.tests.')
    return qname[offset:]

def test(whitelist=None, blacklist=None, cuda=False):
    import sys, os
    from os.path import dirname, join
    import subprocess

    run = failed = 0
    for root, dirs, files in os.walk(join(dirname(__file__), 'tests')):
        qname = qualified_test_name(root)
        exclude_package_dirs(dirs, cuda=cuda)

        for fn in files:
            if fn.startswith('test_') and fn.endswith('.py'):
                if not cuda and fn.startswith('test_cuda_'):
                    continue

                modname, ext = os.path.splitext(fn)
                modname = qname + modname

                if not whitelist_match(whitelist, modname):
                    continue
                if blacklist and whitelist_match(blacklist, modname):
                    continue

                run += 1
                print "running %-60s" % (modname,),
                process = subprocess.Popen([sys.executable, '-m', modname],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT)
                out, err = process.communicate()

                if process.returncode == 0:
                    print "SUCCESS"
                else:
                    print "FAILED: %s" % map_returncode_to_message(
                                                                   process.returncode)
                    print out, err
                    print "-" * 80
                    failed += 1
    
    print "ran test files: failed: (%d/%d)" % (failed, run)
    return failed

def exercise():
    '''Exercise test that are not unittest.TestCase
    '''
    import subprocess
    import sys
    import os
    no_cuda_safe = '''
        test_basic_vectorize.py
        test_parallel_vectorize.py
        test_stream_vectorize.py
        test_gufunc.py
        '''.split()

# These has trouble with the llvm build from CentOS 5
#        test_mini_vectorize.py
#        test_array_expressions.py

    cuda_only = '''
        test_cuda_vectorize.py
        test_cuda_jit.py
        '''.split()

    tests = no_cuda_safe
    try:
        import numbapro._cudadispatch
        tests += cuda_only
    except ImportError: # no cuda?
        pass

    numbapro_dir = os.path.dirname(os.path.realpath(__file__))

    for name in tests:
        print('== Exercising %s' % name)
        script = os.path.join(numbapro_dir, 'tests', name)
        subprocess.check_call([sys.executable, script])
    print('== Exercise completed successfully')

def drop_in_gdb(addr=None, type='int'):
    import os, signal
    if addr is not None:
        print 'watch *(%s *) %s' % (type, hex(addr))
    os.kill(os.getpid(), signal.SIGINT)


import numba
from numba import *
from numbapro.decorators import autojit, jit
from numbapro.parallel.prange import prange
from .vectorize import vectorize, guvectorize

from numba.special import *
from numba.error import *
from numba import typedlist, typedtuple
from numba import (is_registered,
                   register,
                   register_inferer,
                   get_inferer,
                   register_unbound,
                   register_callable)

__all__ = numba.__all__ + ['vectorize', 'guvectorize', 'prange']

import numbapro.cuda
from numbapro.parallel.kernel import CU
