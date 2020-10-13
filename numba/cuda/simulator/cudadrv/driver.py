'''
Most of the driver API is unsupported in the simulator, but some stubs are
provided to allow tests to import correctly.
'''


def device_memset(dst, val, size, stream=0):
    dst.view('u1')[:size].fill(bytes([val])[0])


def host_to_device(dst, src, size, stream=0):
    dst.view('u1')[:size] = src.view('u1')[:size]


def device_to_host(dst, src, size, stream=0):
    host_to_device(dst, src, size)


def device_memory_size(obj):
    return obj.itemsize * obj.size


def device_to_device(dst, src, size, stream=0):
    host_to_device(dst, src, size)


class FakeDriver(object):
    def get_device_count(self):
        return 1


driver = FakeDriver()

Linker = None


def launch_kernel(*args, **kwargs):
    msg = 'Launching kernels directly is not supported in the simulator'
    raise RuntimeError(msg)
