from __future__ import absolute_import, print_function
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import numba
from numba import *
import numbapro._cuda    # import time sideeffect
from numbapro.decorators import autojit, jit
from numbapro.vectorizers import vectorize, guvectorize


def test():
    from numbapro import cuda

    try:
        print('NumbaPro Tests')

        def failfast(ok):
            if not ok:
                raise Exception('Test Failed')

        cfg = dict(buffer=True, verbosity=3)

        print('vectorizers'.center(80, '-'))
        import numbapro.vectorizers.tests.support

        failfast(numbapro.vectorizers.tests.support.run(**cfg))

        print('cuda libraries locator'.center(80, '-'))
        import numba.cuda.cudadrv.libs

        failfast(numba.cuda.cudadrv.libs.test())

        check_cuda()

        if numbapro.cuda.is_available:
            print('cudadrv'.center(80, '-'))
            import numbapro.cudadrv.tests.support

            failfast(numbapro.cudadrv.tests.support.run(**cfg))

            print('cudalib'.center(80, '-'))
            import numbapro.cudalib.tests.support

            failfast(numbapro.cudalib.tests.support.run(**cfg))

            print('cudapy'.center(80, '-'))
            import numbapro.cudapy.tests.support

            failfast(numbapro.cudapy.tests.support.run(**cfg))

            print('cudavec'.center(80, '-'))
            import numbapro.cudavec.tests.support

            failfast(numbapro.cudavec.tests.support.run(**cfg))

        else:
            print('skipped cuda tests')
    except Exception as e:
        import traceback

        traceback.print_exc()
        print('Test failed')
        return False
    else:
        print('All test passed')
        return True


def check_cuda():
    from numba.cuda import is_available, cuda_error

    if not is_available:
        print("CUDA is not available")
        print(cuda_error)
        return

    from numba.cuda.cudadrv import libs
    from numbapro import cuda

    ok = True

    print('libraries detection'.center(80, '-'))
    if not libs.test():
        ok = False

    print('hardware detection'.center(80, '-'))
    if not cuda.detect():
        ok = False

    # result
    if not ok:
        print('FAILED')
    else:
        print('PASSED')
    return ok


def _initialize_all():
    """
    Initialize extensions to Numba
    """
    from numba.npyufunc import Vectorize, GUVectorize

    def init_vectorize():
        from numbapro.cudavec.vectorizers import CudaVectorize
        return CudaVectorize

    def init_guvectorize():
        from numbapro.cudavec.vectorizers import CudaGUFuncVectorize
        return CudaGUFuncVectorize

    Vectorize.target_registry.ondemand['gpu'] = init_vectorize
    Vectorize.target_registry.ondemand['cuda'] = init_vectorize
    GUVectorize.target_registry.ondemand['gpu'] = init_guvectorize
    GUVectorize.target_registry.ondemand['cuda'] = init_guvectorize

_initialize_all()
del _initialize_all

_initialization_completed = True
