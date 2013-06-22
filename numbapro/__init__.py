from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import numba
from numba import *
from numbapro.decorators import autojit, jit
from numbapro.parallel.prange import prange
from numbapro.vectorizers import vectorize, guvectorize

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


def test():
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

    print 'parallel'.center(80, '-')
    import numbapro.parallel.tests.support
    failfast(numbapro.parallel.tests.support.run(**cfg))

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

    else:
        print 'skipped cuda tests'

    print 'All test passed'


