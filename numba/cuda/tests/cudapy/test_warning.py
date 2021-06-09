from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import override_config
from numba.core.errors import NumbaPerformanceWarning
import warnings


@skip_on_cudasim('cudasim does not raise performance warnings')
class TestWarnings(CUDATestCase):
    def test_inefficient_launch_configuration(self):
        @cuda.jit
        def kernel():
            pass

        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
            with warnings.catch_warnings(record=True) as w:
                kernel[1, 1]()

        self.assertEqual(w[0].category, NumbaPerformanceWarning)
        self.assertIn('Grid size', str(w[0].message))
        self.assertIn('2 * SM count', str(w[0].message))

    def test_efficient_launch_configuration(self):
        @cuda.jit
        def kernel():
            pass

        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
            with warnings.catch_warnings(record=True) as w:
                kernel[256, 256]()

        self.assertEqual(len(w), 0)
