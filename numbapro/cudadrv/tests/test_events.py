import numpy as np
from numbapro.cudadrv.old_driver import Event
from numbapro import cuda

import support

@support.addtest
class TestCudaEvent(support.CudaTestCase):
    def test_event_elapsed(self):
        N = 32
        dary = cuda.device_array(N, dtype=np.double)
        evtstart = Event()
        evtend = Event()

        evtstart.record()
        cuda.to_device(np.arange(N), to=dary)
        evtend.record()
        evtend.wait()
        evtend.synchronize()
        print evtstart.elapsed_time(evtend)

    def test_event_elapsed_stream(self):
        N = 32
        stream = cuda.stream()
        dary = cuda.device_array(N, dtype=np.double)
        evtstart = cuda.event()
        evtend = cuda.event()

        evtstart.record(stream=stream)
        cuda.to_device(np.arange(N), to=dary, stream=stream)
        evtend.record(stream=stream)
        evtend.wait(stream=stream)
        evtend.synchronize()
        print evtstart.elapsed_time(evtend)

if __name__ == '__main__':
    support.main()
