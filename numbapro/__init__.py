from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import numba
from numba import *
from numbapro.decorators import autojit, jit
from numbapro.vectorizers import vectorize, guvectorize
from numbapro.cudadrv.initialize import initialize_gpu_target
initialize_gpu_target()
del initialize_gpu_target



# Delay import of CUDA to prevent the CUDA driver from messing with the virtual
# memory space for applications that does not use the GPU feature.
#
#   import numbapro.cuda
#

def test():
    import numbapro.cuda
    try:
        print 'NumbaPro Tests'

        def failfast(ok):
            if not ok:
                raise Exception('Test Failed')
        cfg = dict(buffer=True, verbosity=3)

        print 'npm'.center(80, '-')
        import numbapro.npm.tests.support
        failfast(numbapro.npm.tests.support.run(**cfg))

        print 'vectorizers'.center(80, '-')
        import numbapro.vectorizers.tests.support
        failfast(numbapro.vectorizers.tests.support.run(**cfg))

        print 'cuda libraries locator'.center(80, '-')
        import numba.cuda.cudadrv.libs
        failfast(numba.cuda.cudadrv.libs.test())

        if numbapro.cuda.is_available:
            print 'cudadrv'.center(80, '-')
            import numbapro.cudadrv.tests.support
            failfast(numbapro.cudadrv.tests.support.run(**cfg))

            print 'cudalib'.center(80, '-')
            import numbapro.cudalib.tests.support
            failfast(numbapro.cudalib.tests.support.run(**cfg))

            print 'cudapy'.center(80, '-')
            import numbapro.cudapy.tests.support
            failfast(numbapro.cudapy.tests.support.run(**cfg))

            print 'cudavec'.center(80, '-')
            import numbapro.cudavec.tests.support
            failfast(numbapro.cudavec.tests.support.run(**cfg))

        else:
            print 'skipped cuda tests'
    except Exception, e:
        import traceback
        traceback.print_exc()
        print 'Test failed'
        return False
    else:
        print 'All test passed'
        return True


def check_cuda():
    from numba.cuda.cudadrv import libs
    from numbapro import cuda
    ok = True

    print 'libraries detection'.center(80, '-')
    if not libs.test():
        ok = False

    print 'hardware detection'.center(80, '-')
    if not cuda.detect():
        ok = False

    # result
    if not ok:
        print 'FAILED'
    else:
        print 'PASSED'
    return ok


_initialization_completed = True
