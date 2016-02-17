
def _main(argv, **kwds):
    from numba.testing import run_tests
    # This helper function assumes the first element of argv
    # is the name of the calling program.
    # The 'main' API function is invoked in-process, and thus
    # will synthesize that name.
    return run_tests(argv, defaultTest='numba.tests', **kwds).wasSuccessful()

def main(*argv, **kwds):
    """keyword arguments are accepted for backward compatiblity only.
    See `numba.testing.run_tests()` documentation for details."""

    return _main(['<main>'] + list(argv), **kwds)

if __name__ == '__main__':
    import sys
    # For parallel testing under Windows
    from multiprocessing import freeze_support
    freeze_support()
    sys.exit(0 if _main(sys.argv) else 1)
