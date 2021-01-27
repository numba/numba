'''
The runtime API is unsupported in the simulator, but some stubs are
provided to allow tests to import correctly.
'''


class FakeRuntime(object):
    def get_version(self):
        return (-1, -1)

    def is_supported_version(self):
        return True


runtime = FakeRuntime()
