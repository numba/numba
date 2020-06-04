from numba.core import config
from numba.cuda.cudadrv.runtime import runtime
from numba.cuda.testing import unittest


class TestRuntime(unittest.TestCase):
    def test_get_version(self):
        if config.ENABLE_CUDASIM:
            supported_versions = (-1, -1),
        else:
            supported_versions = ((8, 0), (9, 0), (9, 1), (9, 2), (10, 0),
                                  (10, 1), (10, 2), (11, 0))
        self.assertIn(runtime.get_version(), supported_versions)


if __name__ == '__main__':
    unittest.main()
