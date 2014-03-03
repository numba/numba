from numbapro import cuda
import support, unittest


class TestPinned(support.CudaTestCase):
    def test_profiling(self):
        with cuda._profiling():
            a = cuda.device_array(10)
            del a

        with cuda._profiling():
            a = cuda.device_array(100)
            del a


if __name__ == '__main__':
    unittest.main()

