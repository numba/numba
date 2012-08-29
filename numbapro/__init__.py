
def test(verbosity=2):
    import unittest
    import pkgutil

    test_package_names = ['numbapro.tests.basic_vectorize',
                          'numbapro.tests.llvm_cbuilder',
                          'numbapro.tests.parallel_vectorize',
                          'numbapro.tests.stream_vectorize']

    loader = unittest.TestLoader()
   
    # Find all test scripts in test packages
    test_module_names = []
    for name in test_package_names:
        test_module_names = test_module_names + [name + '.' + module for _,module,_ in pkgutil.iter_modules(__import__(name, fromlist=['']).__path__)]

    suite = loader.loadTestsFromNames(test_module_names)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)
