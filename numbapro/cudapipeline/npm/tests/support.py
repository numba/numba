import unittest, os

TESTS = []

def testcase(func):
    '''Create simple test case from a function and uses the docstring
    as the description of the test.
    '''
    testcase = unittest.FunctionTestCase(func, description=func.__doc__)
    TESTS.append(testcase)
    return testcase

def get_test_suite():
    '''Returns a TestSuite which includes all the tests.
    NOTE: use after discover().
    '''
    suite = unittest.TestSuite()
    for case in TESTS:
        suite.addTest(case)
    return suite

def discover():
    '''Load all testcases in this directory and its subdirectories.
    '''
    scripts = []
    for dirpath, _, filenames in os.walk(os.path.dirname(__file__)):
        for name in filenames:
            if name.startswith('test'):
                modname = name.rsplit('.', 1)[0]
                scripts.append(modname)

    base = __name__.rsplit('.', 1)[0]
    for script in scripts:
        path = '.'.join([base, script])
        __import__(path)

def run(**kws):
    '''
    kws: depends on version of python
        2.6: verbosity[int], descriptions[bool], stream[file]
        2.7: + buffer[bool]
    '''
    discover()
    suite = get_test_suite()
    runner = unittest.TextTestRunner(**kws)

    result = runner.run(suite)

    return result.wasSuccessful()

def main():
    unittest.main()

