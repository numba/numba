import unittest, os, sys

class TestSupport(object):

    def __init__(self, basefile):
        self.basefile = basefile
        self.tests = {}

    def testcase(self, func):
        '''Create simple test case from a function and uses the docstring
        as the description of the test.
        '''
        testcase = unittest.FunctionTestCase(func, description=func.__doc__)
        self.tests[func.__name__] = testcase
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

        base = basedir.rsplit('.', 1)[0].replace(os.path.sep, '.')
        for script in scripts:
            path = '.'.join([base, script])
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
            cfg.pop('buffer', None) # remove unsupported kw
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


def set_base(globals):
    basefile = globals['__file__']
    testsupport = TestSupport(basefile)
    globals['testcase'] = testsupport.testcase
    globals['run'] = testsupport.run
    globals['main'] = testsupport.main
