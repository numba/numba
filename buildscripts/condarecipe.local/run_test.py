import numba.testing
if not numba.testing.multitest():
    raise RuntimeError("Test failed")
print('numba.__version__: %s' % numba.__version__)
