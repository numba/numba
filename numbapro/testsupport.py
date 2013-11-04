import numbapro
import unittest, os, sys, logging
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

logging.basicConfig()

class TestSupport(object):

    def __init__(self, basefile):
        self.basefile = basefile
        self.tests = OrderedDict()

    def testcase(self, func, **kws):
        '''Create simple test case from a function and uses the docstring
        as the description of the test.
        '''
        testcase = unittest.FunctionTestCase(func, description=func.__doc__)
        assert func.__name__ not in self.tests, \
            "duplicated test name %s" % func.__name__
        self.tests[func.__name__] = testcase
        return testcase

    def assertTrue(self, res, msg=""):
        if not res:
            raise AssertionError(msg)

    def assertFalse(self, res, msg=""):
        if res:
            raise AssertionError(msg)

    def addtest(self, testcase):
        assert testcase.__name__ not in self.tests, \
            "duplicated test name %s" % testcase.__name__
        if hasattr(testcase, 'runTest'):
            self.tests[testcase.__name__] = testcase()
        else:
            for meth in dir(testcase):
                if meth.startswith('test'):
                    fullname = '%s.%s' % (testcase.__name__, meth)
                    self.tests[fullname] = testcase(meth)
        return testcase

    def get_test_suite(self):
        '''Returns a TestSuite which includes all the tests.
        NOTE: use after discover().
        '''
        suite = unittest.TestSuite(self.tests.values())
        return suite

    def discover(self):
        '''Load all testcases in this directory and its subdirectories.
        '''
        scripts = []
        basedir = os.path.dirname(self.basefile)

        for dirpath, _, filenames in os.walk(basedir):
            for name in filenames:
                if name.startswith('test'):
                    modname = name.rsplit('.', 1)[0]
                    scripts.append(modname)

        package_base = os.path.dirname(numbapro.__file__)
        assert basedir.startswith(package_base), \
                    "not relative sub-directory to base"
        base = basedir[len(package_base) + 1:]
        base = base.rsplit('.', 1)[0].replace(os.path.sep, '.')

        for script in scripts:
            path = '.'.join(['numbapro', base, script])
            __import__(path)

    def run(self, **kws):
        '''
        kws: depends on version of python
            2.6: verbosity[int], descriptions[bool], stream[file]
            2.7: + buffer[bool]
            
        NOTE: for use to run the entire test suite.
        '''
        self.discover()
        suite = self.get_test_suite()
        
        if sys.version_info[:2] <= (2, 6):
            kws.pop('buffer', None) # remove unsupported kw
        runner = unittest.TextTestRunner(**kws)

        result = runner.run(suite)

        return result.wasSuccessful()

    def main(self):
        '''
        NOT: for use to run a single test script.
        '''
        cfg = dict(verbosity=3, descriptions=True)
        runner = unittest.TextTestRunner(**cfg)
        tests = [self.tests[name] for name in sys.argv[1:]]
        if tests:
            suite = unittest.TestSuite(tests)
        else:
            suite = self.get_test_suite()
        result = runner.run(suite)
        return result


def set_base(globals):
    basefile = globals['__file__']
    testsupport = TestSupport(basefile)
    globals['testcase'] = testsupport.testcase
    globals['addtest'] = testsupport.addtest
    globals['run'] = testsupport.run
    globals['main'] = testsupport.main
    globals['assertTrue'] = testsupport.assertTrue
    globals['assertFalse'] = testsupport.assertFalse
