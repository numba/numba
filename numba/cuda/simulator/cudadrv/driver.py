'''
Most of the driver API is unsupported in the simulator, but some stubs are
provided to allow tests to import correctly.
'''

host_to_device = None
device_to_host = None

class FakeDriver(object):
    def get_device_count(self):
        return 1

driver = FakeDriver()

Linker = None
