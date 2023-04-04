
from numba import cuda
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
                                skip_unless_cc_53, skip_unless_cudasim)
from numba.core import config
from numba.tests.support import override_config


class TestIsFP16Supported(CUDATestCase):
    def setUp(self):
        super().setUp()
        self._use_cuda_bindings = config.CUDA_USE_NVIDIA_BINDING

        # Default disable cuda binding
        config.CUDA_USE_NVIDIA_BINDING = 0

    def tearDown(self):
        super().tearDown()
        config.CUDA_USE_NVIDIA_BINDING = self._use_cuda_bindings

    @skip_on_cudasim
    @skip_unless_cc_53
    def test_is_fp16_supported(self):
        self.assertFalse(cuda.is_float16_supported())
        with override_config('CUDA_USE_NVIDIA_BINDING', 1):
            self.assertTrue(cuda.is_float16_supported())

    @skip_unless_cudasim
    def test_is_fp16_supported_on_simulator(self):
        self.assertTrue(cuda.is_float16_supported())

    @skip_on_cudasim
    @skip_unless_cc_53
    def test_device_supports_float16(self):
        self.assertTrue(cuda.get_current_device().supports_float16)
