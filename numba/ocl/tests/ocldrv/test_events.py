from __future__ import absolute_import, print_function
import numpy as np
from numba import ocl
import unittest


class TestOpenCLEvent(unittest.TestCase):
    @unittest.skip("equivalent to cuda event not yet implemented")
    def test_event_elapsed(self):
        N = 32
        dary = ocl.device_array(N, dtype=np.double)
        evtstart = ocl.event()
        evtend = ocl.event()

        evtstart.record()
        ocl.to_device(np.arange(N), to=dary)
        evtend.record()
        evtend.wait()
        evtend.synchronize()
        print(evtstart.elapsed_time(evtend))

    @unittest.skip("equivalent to cuda event not yet implemented")
    def test_event_elapsed_stream(self):
        N = 32
        stream = ocl.stream()
        dary = ocl.device_array(N, dtype=np.double)
        evtstart = ocl.event()
        evtend = ocl.event()

        evtstart.record(stream=stream)
        ocl.to_device(np.arange(N), to=dary, stream=stream)
        evtend.record(stream=stream)
        evtend.wait(stream=stream)
        evtend.synchronize()
        print(evtstart.elapsed_time(evtend))

if __name__ == '__main__':
    unittest.main()
