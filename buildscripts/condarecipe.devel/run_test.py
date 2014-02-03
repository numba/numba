import numba
if not numba.test():
    raise RuntimeError("Test failed")
print('numba.__version__: %s' % numba.__version__)
