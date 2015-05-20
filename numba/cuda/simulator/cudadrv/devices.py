class FakeCUDAContext(object):
    '''
    This stub implements functionality only for simulating a single GPU
    at the moment.
    '''
    def __init__(self, device):
        self._device = device

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
       pass

    def __str__(self):
        return "<Managed Device {self.id}>".format(self=self)

    @property
    def id(self):
        return self._device

    @property
    def compute_capability(self):
        return (5, 2)


class FakeDeviceList(object):
    '''
    This stub implements a device list containing a single GPU. It also
    keeps track of the GPU status, i.e. whether the context is closed or not,
    which may have been set by the user calling reset()
    '''
    def __init__(self):
        self.lst = (FakeCUDAContext(0),)
        self.closed = False

    def __getitem__(self, devnum):
        self.closed = False
        return self.lst[devnum]

    def __str__(self):
        return ', '.join([str(d) for d in self.lst])

    def __iter__(self):
        return iter(self.lst)

    def __len__(self):
        return len(self.lst)

    @property
    def current(self):
        if self.closed:
            return None
        return self.lst[0]


gpus = FakeDeviceList()


def reset():
    gpus[0].closed = True


def require_context(func):
    '''
    In the simulator, a context is always "available", so this is a no-op.
    '''
    return func
