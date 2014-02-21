from numbapro.cudadrv.old_driver import Driver, CudaDriverError
from .support import main, testcase

@testcase
def test_bad_driver():
    drv = Driver()
    try:
        drv.easteregg()
    except CudaDriverError:
        pass
    else:
        raise AssertionError('CudaDriverError not raised')


if __name__ == '__main__':
    main()
